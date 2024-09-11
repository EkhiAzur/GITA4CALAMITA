# GITA4CALAMITA

The `GITA4CALAMITA` repository is designed to evaluate language models on various tasks related to story comprehension and conflict detection. For more detailed instructions and examples, please refer to the individual scripts and their docstrings in the repository. This README provides instructions on how to set up, run, and evaluate the benchmarks.

## Tasks

The benchmark consists of the following tasks in the harness format in the task format. Tasks are defined in the `tasks` folder. Each task folder contains a yaml config file with the harness config and a utils file with the helper functions for the prompt generation. The general task description is included in the config file and is only included at the start of the prompt. The utils file includes the prompt for each instance. All tasks are evaluated in a 3-shot setting, using random examples from the test set. For instruct models, the description is included in the system prompt. An each instance is formatted as a multiturn conversation between user and assistant. 

1. **Story Classification:** Plausible? True or False
2. **Conflict Detection:** Which sentence pair creates the conflict? [0-4]
3. **Physical State:** Which physical state is the cause of the conflict? Open, Hot-Cold, etc.

## Task Prompt Examples

### Story Classification

```text
Please read the following story and answer if the story is plausible taking into account the order of the events. Please answer with true or false.

The story is as follows:
Ho aperto il sito dell'università C'era il bando che mi interessava. Il bando era chiuso. Ho inserito i miei dati. La mia candidatura è stata accettata. 
Is the story plausible? false

The story is as follows:
Giuseppe ha comprato il quotidiano. Giuseppe legge il quotidiano. Giuseppe fa a pezzi il quotidiano. Giuseppe brucia i pezzi del quotidiano. Giuseppe ha una stufa. 
Is the story plausible? true

The story is as follows:
Giogio ha comprato uno spremiagrumi automatico. Giorgio vuole fare un succo di arance. Giorgio non ha arance. Giorgio ha spremuto le arance. Giorgio ha bevuto il succo. 
Is the story plausible? false

The story is as follows:
Marco ha aperto il frigo. Marco ha preso il latte. Marco ha preso la tazza. Marco ha preso il cucchiaio. Marco ha messo il cucchiaio nella tazza. 
Is the story plausible?
```

### Conflict Detection

```text
The following story is implausible. Identify the breakpoint, and then select the sentence responsible for the implausibility. Please identify the breakpoint sentence and the conflicting sentence.

The story is as follows: 
0. Giorgio ha aperto la finestra.
1. Giorgio ha chiuso la finestra.
2. Giorgio ha messo il tappeto sul davanzale.
3. Giorgio ha sbattuto il tappeto.
4. Giorgio ha raccolto il tappeto del salone.

The conflicting sentence and the breakpoint are: 1 and 2

The story is as follows: 
0. È appena arrivata la nuova cyclette.
1. Giulia ha sistemato la cyclette in salone.
2. Giulia ha usato la cyclette.
3. La cyclette funziona.
4. Giulia ha azionato la cyclette.

The conflicting sentence and the breakpoint are: 2 and 3

The story is as follows: 
0. Alberto prende il caffè dalla dispensa.
1. Alberto riempie la caffettiera di caffè.
2. Alberto apre la caffettiera.
3. Alberto chiude la caffettiera.
4. Alberto mette la caffettiera sul fuoco.

The conflicting sentence and the breakpoint are: 1 and 2

The story is as follows: 
0. Marco ha preso il latte.
1. Marco ha aperto il frigo.
2. Marco ha preso la tazza.
3. Marco ha preso il cucchiaio.
4. Marco ha messo il cucchiaio nella tazza.

The conflicting sentence and the breakpoint are:
```

### Physical State

```text
The following story is implausible. Identify the physical state that causes the conflict in the story. These are the descriptions of each physical state: 
Power: Indicates whether an object is powered or not, relevant for electrical devices. 
Location: Refers to the spatial position of an entity, either human or object. 
Exist: Denotes whether an object is present or has disappeared. 
Clean: Refers to the cleanliness of an entity, indicating whether it is clean or dirty. 
Edible: Identifies whether an object is fit for consumption. 
Wet: Denotes whether an object or person is in a wet or dry state. 
Functional: Refers to whether an object is in working condition or broken. 
Wearing: Applies to humans, indicating whether they are dressed or not. 
Open: Refers to whether an object (e.g., a door or container) is open or closed. 
Conscious: Denotes whether a human is conscious or unconscious. 
Temperature: Refers to the relative temperature of an entity, e.g., hot or cold. 
Solid: Describes whether an object is in a solid state. 
Occupied: Indicates whether an object (e.g., a container) is occupied or contains something. 
In pieces: Refers to whether an object is intact or has been broken into pieces. Select one of them after reading the story.

The story is as follows: 
Filomena ha chiamato il tecnico. La caldaia si è rotta. Il tecnico non è ancora venuto a riparare la caldaia. Filomena non ha fatto una doccia calda. Filomena è andata a letto presto. 
The physical state that causes the conflict in the implausible story is: functional

The story is as follows: 
Jaime ha preso il casco. Jaime è uscito da casa. La vespa di Jaime è stata rubata. Jaime è andato al lavoro in vespa. Jaime è arrivato al lavoro. 
The physical state that causes the conflict in the implausible story is: exist

The story is as follows: 
La professoressa di arte arriva in aula. La professoressa spacca il proiettore. La professoressa prende le diapositive e inizia a proiettarle sulla parete. Le diapositive si vedono bene. La professoressa finisce dopo un'ora. 
The physical state that causes the conflict in the implausible story is: in pieces

The story is as follows: 
Marco ha preso il latte. Marco ha aperto il frigo. Marco ha preso la tazza. Marco ha preso il cucchiaio. Marco ha messo il cucchiaio nella tazza. 
The physical state that causes the conflict in the implausible story is:
```

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/EkhiAzur/GITA4CALAMITA.git
    cd GITA4CALAMITA
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv harness_env
    source harness_env/bin/activate
    ```

3. **Install the dependencies:**
    ```sh
    cd lm-evaluation-harness
    pip install git+https://github.com/juletx/lm-evaluation-harness
    ```

## Unified Evaluation

Use the `lm-evaluation-harness` to run the evaluation. The `unified_code.py` script is a modified script of harness that allows to run evaluations of the three tasks in a dependent way. The models are evaluated in the instances that are only guessed correctly in the previous tasks. The script also allows to evaluate instruct models.

Example command to evaluate the model `Meta-Llama-3.1-8B` included in the `Llama3_1-8b.slurm` file.

```sh
path="meta-llama"
model="Meta-Llama-3.1-8B"
model_name=$path/$model
num_fewshot=3
tasks_path="../tasks"

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --batch_size auto:64 \
    --device cuda:0 \
    --num_fewshot ${num_fewshot} \
    --output_path ../results/${model} \
    --include_path ${tasks_path} \
```

To evaluate instruct models include the additional chat arguments. Example command to evaluate the model `Meta-Llama-3.1-8B-Instruct` included in the `Llama3_1-8b-it.slurm` file.

```sh
path="meta-llama"
model="Meta-Llama-3.1-8B-Instruct"
model_name=$path/$model
num_fewshot=3
tasks_path="../tasks"

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --batch_size auto:64 \
    --device cuda:0 \
    --num_fewshot ${num_fewshot} \
    --output_path ../results/${model} \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path ${tasks_path} \
```

## Unified Evaluation Results

View the results:
    - The results will be saved in the `results/` directory.
    - You can view the results in the JSON files generated.

|      Metric       |n-shot|   Metric value    |
|-------------------|-----:|------------------:|
|Accuracy           |     3|0.60955056179775280|
|Consistency        |     3|0.26470588235294120|
|Verifiability      |     3|0.11344537815126050|
|Cloze Accuracy     |     3|0.70434782608695660|
|Order Accuracy     |     3|0.60655737704918030|
|Cloze Consistency  |     3|0.33043478260869563|
|Order Consistency  |     3|0.20661157024793390|
|Cloze Verifiability|     3|0.13043478260869565|
|Order Verifiability|     3|0.09917355371900827|

|      Metric       |n-shot|    Metric value    |
|-------------------|-----:|-------------------:|
|Accuracy           |     3|0.772471910112359600|
|Consistency        |     3|0.373949579831932800|
|Verifiability      |     3|0.105042016806722690|
|Cloze Accuracy     |     3|0.939130434782608700|
|Order Accuracy     |     3|0.901639344262295100|
|Cloze Consistency  |     3|0.539130434782608700|
|Order Consistency  |     3|0.223140495867768600|
|Cloze Verifiability|     3|0.165217391304347820|
|Order Verifiability|     3|0.049586776859504134|