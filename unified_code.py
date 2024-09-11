import jsonlines
import argparse
import logging
import os
from functools import partial
import shutil
import gc
import torch
import json

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string, handle_non_serializable

def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(
            default_items[num_items:]
        )  # extend items list with missing defaults

    return items

def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the evaluation script
    Copied from lm-harness
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=True,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default="tasks",
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Name of the run",
    )

    return parser


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()

def prepare_env(args):
    """
    Prepare the environment for evaluation, creation of loggers, and setting up the evaluation tracker
    Copied from lm-harness
    """

    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))
    else:
        wandb_logger = None

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "If fewshot_as_multiturn is set, apply_chat_template must be set to True."
        )

    if (
        args.num_fewshot is None or args.num_fewshot == 0
    ) and args.fewshot_as_multiturn:
        raise ValueError(
            "If fewshot_as_multiturn is set, num_fewshot must be greater than 0."
        )

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        args.model_args = args.model_args + ",trust_remote_code=True"

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests
    )    

    return evaluation_tracker, request_caching_args, wandb_logger, eval_logger

def log_results(results, args=None, wandb_logger=None, evaluation_tracker=None, eval_logger=None):

    """
    Log the results to the evaluation tracker and Wandb
    Copied from lm-harness
    """

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)


        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

def prepare_task(task, args):

    """
    Prepare the task for evaluation. The tasks must be in the ./tasks directory, following the structure of the lm_eval tasks.
    Copied and adapted from lm-harness
    """

    task = os.path.join(args.include_path, task)

    model_name = get_model_name(args)

    if os.path.isdir(task):
        import glob

        task_names = []
        yaml_path = os.path.join(task, "*.yaml")
        for yaml_file in glob.glob(yaml_path):
            config = utils.load_yaml_config(yaml_file)

            if "dataset_kwargs" in config and "data_files" in config["dataset_kwargs"]:
                config["dataset_kwargs"]["data_files"]["test"] = os.path.join(f".eval_aux_{model_name}", config["dataset_kwargs"]["data_files"]["test"])

            task_names.append(config)
    else:
        raise ValueError(f"Task {task} not found")

    return task_names

def get_well_predicted_samples(results_per_sample, task_name):

    """
    Collect well predicted samples to use in the next task evaluation
    In case of story_class, only implausible cases are considered
    """

    well_predicted_samples = []

    for sample in results_per_sample:

        if int(sample["acc"]) == 1:

            if (task_name == "story_class") and (not sample["doc"]["plausible"]): # Only well predicted implausible cases are considered to use in conflict_detec

                    well_predicted_samples.append(sample["doc"])

                    continue
        
            well_predicted_samples.append(sample["doc"])

    return well_predicted_samples

def eval_task(task_name, task_path, args, task_manager, evaluation_tracker, request_caching_args):

    """
    Function to evaluate a task, it prepares the task, evaluates it, collects well predicted samples and returns the results
    """

    task = prepare_task(task_path, args)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        **request_caching_args,
    )

    well_predicted_cases = get_well_predicted_samples(results["samples"][task_name], task_name)

    return results, well_predicted_cases

def prepare_task_dataset(task, args, selected_instances = None):
    """
    Function to prepare the task dataset for evaluation only if selected_instances is not None
    """

    model_name = get_model_name(args)

    if selected_instances is not None:
        jsonlines.Writer(open(f".eval_aux_{model_name}/{task}.jsonl", 'w')).write_all(selected_instances)

def run_taks_eval(task, args, task_manager, evaluation_tracker, request_caching_args, previous_well_predicted_cases = None):

    """
    Main function to run the task evaluation, it prepares the task dataset, evaluates the task and returns the results and well predicted cases
    """

    prepare_task_dataset(task, args, selected_instances=previous_well_predicted_cases)

    task_path = task_manager.match_tasks(f"{task}")[0]

    results, well_predicted_cases = eval_task(task, task_path, args, task_manager, evaluation_tracker, request_caching_args)

    return results, well_predicted_cases


def make_table(result_dict, n_shot):

    """
    Function to create a markdown table from the results
    Adapted from lm-harness.utils.make_table
    """

    from pytablewriter import MarkdownTableWriter

    all_headers = [
        "Metric",
        "n-shot",
        "Metric value",
    ]

    md_writer = MarkdownTableWriter()
    md_writer.headers = all_headers

    values = []

    for metric_name, metric_value in result_dict.items():

        values.append([metric_name, n_shot, metric_value])

    md_writer.value_matrix = values

    return md_writer.dumps()

def story_class_acc_cloze_order_plausible(task_results, task_name):

    """
    Function to calculate the cloze and order accuracy
    """

    cloze_num = 0
    order_num = 0
    plausible_num = 0

    cloze_well = 0
    order_well = 0
    plausible_well = 0

    implausible_cloze_num = 0
    implausible_order_num = 0
    plausible_num = 0

    for sample in task_results["samples"][task_name]:

        if "O" in sample["doc"]["example_id"]:
            order_num += 1

            if not sample["doc"]["plausible"]:
                implausible_order_num += 1

            if int(sample["acc"]) == 1:
                order_well += 1
            
            continue

        elif "C" in sample["doc"]["example_id"]:
            cloze_num += 1

            if not sample["doc"]["plausible"]:
                implausible_cloze_num += 1

            if int(sample["acc"]) == 1:
                cloze_well += 1
            
            continue
    
        else:

            plausible_num += 1

            if int(sample["acc"]) == 1:
                plausible_well += 1


    return cloze_well / cloze_num, order_well / order_num, implausible_cloze_num, implausible_order_num, plausible_well / plausible_num

def count_cloze_order(task_results, task_name):

    """
    Function to count the well predicted cloze and order cases
    """

    well_cloze_num = 0
    well_order_num = 0

    for sample in task_results["samples"][task_name]:

        if "O" in sample["doc"]["example_id"]:

            if int(sample["acc"]) == 1:

                well_order_num += 1

        if "C" in sample["doc"]["example_id"]:

            if int(sample["acc"]) == 1:
                well_cloze_num += 1

    return well_cloze_num, well_order_num

def calculate_cloze_order_metrics(story_class_results, conflict_detec_results, physical_state_results):

    """
    Function to calculate separated cloze and order metrics
    """

    ### story_class ###

    cloze_acc, order_acc, implausible_cloze_num, implausible_order_num, plausible_acc = story_class_acc_cloze_order_plausible(story_class_results, "story_class")

    ### conflict_detec ###

    well_cloze_conflict_detec_num, well_order_conflict_detec_num = count_cloze_order(conflict_detec_results, "conflict_detec")

    ### physical_state ###

    well_cloze_physical_state_num, well_order_physical_state_num = count_cloze_order(physical_state_results, "physical_state")

    ### Calculate metrics ###

    return {
        "Cloze Accuracy": cloze_acc,
        "Order Accuracy": order_acc,
        "Plausible Accuracy": plausible_acc,
        "Cloze Consistency": well_cloze_conflict_detec_num / implausible_cloze_num,
        "Order Consistency": well_order_conflict_detec_num / implausible_order_num,
        "Cloze Verifiability": well_cloze_physical_state_num / implausible_cloze_num,
        "Order Verifiability": well_order_physical_state_num / implausible_order_num,
    }

def count_implausible_cases(story_class_results):

    """
    Function to count the implausible cases in story_class
    """

    implausible_cases = 0

    for sample in story_class_results["samples"]["story_class"]:

        if not sample["doc"]["plausible"]:
            implausible_cases += 1

    return implausible_cases

def select_non_plausible_cases(cases_list):

    """
    Function to select the non plausible cases from a list of cases
    """

    non_plausible_cases = []

    for case in cases_list:

        if not case["plausible"]:
            non_plausible_cases.append(case)

    return non_plausible_cases

def get_model_name(args):

    # We assume that the model name is the first argument in the model_args

    return args.model_args.split(",")[0].split("=")[1].split("/")[1]

def main(args):

    model_name = get_model_name(args)

    if not os.path.exists(f".eval_aux_{model_name}"):
        os.makedirs(f".eval_aux_{model_name}")

    metrics = {}

    evaluation_tracker, request_caching_args, wandb_logger, eval_logger = prepare_env(args)

    task_manager = TaskManager(args.verbosity, include_path=args.include_path, include_defaults=False)

    ### Story classification ### story_class

    story_class_results, story_class_well_predicted_cases = run_taks_eval("story_class", args, task_manager, evaluation_tracker, request_caching_args, previous_well_predicted_cases=None)
    
    # Count implausible cases to calculate consistency and verifiability
        
    implausible_cases = count_implausible_cases(story_class_results)

    torch.cuda.empty_cache()
    gc.collect()

    if len(story_class_well_predicted_cases) == 0: # 0% acc do not calculate other metrics

        metrics["Accuracy"] = 0.0
        metrics["Consistency"] = 0.0
        metrics["Verifiability"] = 0.0

        print(make_table(metrics, args.num_fewshot))
        return 0, 0, 0

    non_plausible_story_class_well_predicted = select_non_plausible_cases(story_class_well_predicted_cases)

    story_class_acc= story_class_results["results"]["story_class"]["acc,none"]

    metrics["Accuracy"] = story_class_acc

    ### Conflict detection ### conflict_detec

    conflict_detec_results, conflict_detec_well_predicted_cases = run_taks_eval("conflict_detec", args, task_manager, evaluation_tracker, request_caching_args, previous_well_predicted_cases=non_plausible_story_class_well_predicted)

    torch.cuda.empty_cache()
    gc.collect()

    consistency = len(conflict_detec_well_predicted_cases) / implausible_cases

    metrics["Consistency"] = consistency
    
    ### Physical state classification ### physical_state

    physical_state_results, physical_state_well_predicted_cases = run_taks_eval("physical_state", args, task_manager, evaluation_tracker, request_caching_args, previous_well_predicted_cases=conflict_detec_well_predicted_cases)

    torch.cuda.empty_cache()
    gc.collect()

    verifiability = len(physical_state_well_predicted_cases) / implausible_cases

    metrics["Verifiability"] = verifiability

    ### Calculate Cloze and Order metrics ###

    cloze_order_results = calculate_cloze_order_metrics(story_class_results, conflict_detec_results, physical_state_results)

    metrics.update(cloze_order_results)

    table = make_table(metrics, args.num_fewshot)

    print(table)

    shutil.rmtree(f".eval_aux_{model_name}")

    
    story_class_results["story_class"] = {
      "Accuracy": story_class_acc,
      "alias": "story_class"
    }

    conflict_detec_results["conflict_detec"] = {
        "Consistency": consistency,
        "alias": "conflict_detec"
    }

    physical_state_results["physical_state"] = {
        "Verifiability": verifiability,
        "alias": "physical_state"
    }

    log_results(story_class_results, args, wandb_logger, evaluation_tracker, eval_logger)

    log_results(conflict_detec_results, args, wandb_logger, evaluation_tracker, eval_logger)

    log_results(physical_state_results, args, wandb_logger, evaluation_tracker, eval_logger)

    batch_sizes = "story_class: "+",".join(map(str, story_class_results["config"]["batch_sizes"]))+"\n"
    batch_sizes += "conflict_detec: "+",".join(map(str, conflict_detec_results["config"]["batch_sizes"]))+"\n"
    batch_sizes += "physical_state: "+",".join(map(str, physical_state_results["config"]["batch_sizes"]))

    print(
        f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
        f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )

    metric_file_path = os.path.join(args.output_path, "metrics.txt")

    with open(metric_file_path, "w") as f:
        f.write(table)

    if args.wandb_args:
        # Tear down wandb run once all the logging is done.
        wandb_logger.run.finish()

    return metrics

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    args = parse_eval_args(setup_parser())

    results = main(args)

    for metric, value in results.items():
        logging.info(f"{metric}: {value}")