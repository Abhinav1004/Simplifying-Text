# Import relevant libraries
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util.evaluate_model.temp.cli import evaluate_system_output
from contextlib import contextmanager
import torch
from util.processing.preprocessor import get_data_filepath, yield_lines, read_lines, count_line, write_lines
from util.evaluate_model.temp.sari import corpus_sari
import time
from util.simsum_models.t5_simsum import SumSim


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()

# set random seed universal
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
model_dir = None
_model_dirname = None
max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify the model_name and checkpoint_name

model_dirname = 'exp_WikiDocFinal_T5_kw_num4_div0.9_0.001CosSim(ReLU_W)'
checkpoint_path = 'checkpoint-epoch=5.ckpt'

#### Single model ####
# load the model
# Model = BartBaseLineFineTuned.load_from_checkpoint(EXP_DIR / model_dirname / checkpoint_path).to(device)
# model = Model.model.to(device)
# tokenizer = Model.tokenizer
#### Single model ####


#### Joint model ####
Model = SumSim.load_from_checkpoint(EXP_DIR /  model_dirname / checkpoint_path).to(device)
summarizer = Model.summarizer.to(device)
simplifier = Model.simplifier.to(device)
summarizer_tokenizer = Model.summarizer_tokenizer
simplifier_tokenizer = Model.simplifier_tokenizer
#### Joint model ####

def generate_single(sentence, preprocessor = None):
    '''
    This function is for T5 or Bart single model to generate/predict
    '''

    # text = "simplify: " + sentence  ### -> for T5
    text = sentence
    encoding = tokenizer(text, max_length=512,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        num_beams=2,
        top_k=70,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1,
    )
    sent = tokenizer.decode(beam_outputs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent


def generate(sentence, preprocessor=None):
    '''
    This function is for Joint model to generate/predict
    '''

    # For T5
    sentence = 'summarize: ' + sentence

    encoding = summarizer_tokenizer(
        [sentence],
        max_length = 512,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'pt',
    )

    summary_ids = summarizer.generate(
        encoding['input_ids'].to(device),
        num_beams = 15,
        min_length = 30,
        max_length = 512,
        top_k = 80, top_p = 0.97
    ).to(device)

    # For T5
    for i, summary_id in enumerate(summary_ids):
        add_tokens = torch.tensor([18356, 10]).to(device)
        summary_ids[i,:] = torch.cat((summary_id, add_tokens), dim=0)[:-2]

    summary_atten_mask = torch.ones(summary_ids.shape).to(device)
    summary_atten_mask[summary_ids[:,:] == summarizer_tokenizer.pad_token_id] = 0

    beam_outputs = simplifier.generate(
        input_ids = summary_ids,
        attention_mask = summary_atten_mask,
        do_sample = True,
        max_length = 256,
        num_beams = 5, #16
        top_k = 80,  #120
        top_p = 0.95, #0.95
        early_stopping = True,
        num_return_sequences = 1,
    )

    sent = simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent

    '''
    sentence = preprocessor.encode_sentence(sentence)
    text = "simplify: " + sentence
    encoding = tokenizer(text, max_length=max_len,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    # set top_k = 130 and set top_p = 0.97 and num_return_sequences = 1
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_len,
        num_beams=10,
        top_k=130,
        top_p=0.97,
        early_stopping=True,
        num_return_sequences=1
    )
    pred_sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return pred_sent
    # final_outputs = []
    # for beam_output in beam_outputs:
    #     sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     if sent.lower() != sentence.lower() and sent not in final_outputs:
    #         final_outputs.append(sent)

    # return final_outputs
    '''

def evaluate(orig_filepath, sys_filepath, ref_filepaths):
    orig_sents = read_lines(orig_filepath)
    # NOTE: change the refs_sents if several references are used
    refs_sents = [read_lines(ref_filepaths)]
    #refs_sents = [read_lines(filepath) for filepath in ref_filepaths]

    return corpus_sari(orig_sents, read_lines(sys_filepath), refs_sents)

def back_translation(text):
    X = translator.translate(text, dest = 'de')
    return translator.translate(X.text, dest = 'en').text


def simplify_file(complex_filepath, output_filepath, features_kwargs=None, model_dirname=None, post_processing=True):
    '''
    Obtain the simplified sentences (predictions) from the original complex sentences.
    '''

    total_lines = count_line(complex_filepath)
    print(complex_filepath)
    print(complex_filepath.stem)

    output_file = Path(output_filepath).open("w")

    for n_line, complex_sent in enumerate(yield_lines(complex_filepath), start=1):
        ### NOTE: Change it when using Single model or Joint model
        #output_sents = generate_single(complex_sent, preprocessor = None)
        output_sents = generate(complex_sent, preprocessor=None)


        print(f"{n_line+1}/{total_lines}", " : ", output_sents)
        if output_sents:
            # output_file.write(output_sents[0] + "\n")
            output_file.write(output_sents + "\n")
        else:
            output_file.write("\n")
    output_file.close()

    if post_processing: post_process(output_filepath)

def post_process(filepath):
    lines = []
    for line in yield_lines(filepath):
        lines.append(line.replace("''", '"'))
    write_lines(lines, filepath)


def evaluate_on_WIKIDOC(phase, features_kwargs=None,  model_dirname = None):
    dataset = WIKI_DOC
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)
    #features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{dataset}_{phase}.log.txt'
    complex_filepath =get_data_filepath(dataset, phase, 'complex_kw_num4_div0.9')# _kw_num3_div0.9'

    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath =get_data_filepath(dataset, phase, 'complex_kw_num4_div0.9')

        #complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            scores  = evaluate_system_output(test_set='custom',
                                             sys_sents_path=str(pred_filepath),
                                             orig_sents_path=str(complex_filepath),
                                             refs_sents_paths=str(ref_filepaths))


            print("SARI: {:.2f}\t D-SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['D-sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

def evaluate_on_D_WIKI(phase, features_kwargs=None,  model_dirname = None):
    dataset = D_WIKI

    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)
    #features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{dataset}_{phase}.log.txt'
    complex_filepath =get_data_filepath(dataset, phase, 'complex') #_kw_num3_div0.7

    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath =get_data_filepath(dataset, phase, 'complex')

        #complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            scores  = evaluate_system_output(test_set='custom',
                                             sys_sents_path=str(pred_filepath),
                                             orig_sents_path=str(complex_filepath),
                                             refs_sents_paths=str(ref_filepaths))


            print("SARI: {:.2f}\t D-SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['D-sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))



####### WIKI_DOC #######
evaluate_on_WIKIDOC(phase='test', features_kwargs=None, model_dirname=model_dirname)




####### D_WIKI #######
#evaluate_on_D_WIKI(phase='test', features_kwargs=None, model_dirname=model_dirname)
