from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time
import utils
import os
import csv
import itertools
import argparse
import sys
from sklearn.model_selection import ParameterGrid
import pickle as pkl
import numpy as np


class Notes:
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        corpus_path = os.path.join(utils.data_path, self.corpus_file)
        count = 0
        with open(corpus_path, 'r', encoding='utf8') as f:
            rd = csv.reader(f, delimiter=',')
            for line in rd:
                count += 1
                # assume there's one document per line, tokens separated by whitespace
                yield [w for w in line]


# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0
        self.loss_previous_step = 0.0
        self.latest_training_loss = 0.0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.loss_previous_step += float(loss)
        self.epoch += 1
        # if self.epoch % 10 == 0:
        #     print(f'Loss after epoch {self.epoch}: {loss} -- Cumulative loss: {self.loss_previous_step}')
        if self.epoch != self.epochs:
            model.running_training_loss = 0.0


def train_embeddings(corpus_file, param, model_name, iter=1):
    sentences = Notes(corpus_file)
    print("Finished loading notes\n")
    start = time.time()
    col_results = [
        ("epochs", "start_alpha", "end_alpha", "vector_size", "min_count", "window", "PP_mean", "PP_sd", "w2v")]
    results = []
    list_par = list(ParameterGrid(param))
    i = 0
    for par in list_par:
        pp_tmp = []
        mod_tmp = []
        for s in range(iter):
            start = time.time()
            w2v = Word2Vec(vector_size=par['vector_size'], window=par['window'], sg=0, hs=1, min_count=par['min_count'],
                           workers=8, seed=s, negative=0)
            w2v.build_vocab(sentences)
            w2v.train(sentences,
                      compute_loss=True, epochs=par['epochs'],
                      callbacks=[callback(epochs=par['epochs'])], start_alpha=par['start_alpha'],
                      end_alpha=par['end_alpha'], total_examples=w2v.corpus_count)
            wvec = w2v.wv
            # pp_vect, pp_mean, pp_sd = evaluate(sentences, w2v, wvec.index_to_key)
            pp = evaluate(sentences, w2v, wvec.index_to_key)
            pp_tmp.append(pp)
            mod_tmp.append(w2v)
        results.append(
            (par["epochs"], par["start_alpha"], par["end_alpha"], par["vector_size"], par["min_count"],
             par["window"], np.mean(pp_tmp), np.std(pp_tmp), mod_tmp))
        i += 1
        print(f"Parameter combination {i}/{len(list_par)} finished in {round(time.time() - start, 2)}s")
    results = sorted(results, key=lambda x: x[-3], reverse=False)
    pkl.dump(col_results + [r[:-1] for r in results],
             open(os.path.join(utils.data_path, f'{model_name}_gridsearch.pkl'), 'wb'))
    print(f"Training on notes ended in {round(time.time() - start, 2)}s")
    models = results[0][-1]
    models[0].save(os.path.join(utils.data_path, model_name))
    print(f"Vocabulary size: {len(models[0].wv.index_to_key)}")
    print(f"Best hyperparameters selected based on PP score:")
    for idx, c in enumerate(col_results[0]):
        print(f"{c}: {results[0][idx]}")
    print('\n')
    # print(f"Perplexity on training from loss: {2**model.get_latest_training_loss()}")
    return models, models[0].wv.index_to_key


def evaluate(sentences, model, vcb):
    """
    Evaluate word2vec model on test set sentences and return perplexity
    :param ts_sentences: list of tokenized sentences
    :param model: trained model
    :return: perplexity for sentences
    """
    ts_sentences = [[w for w in s if w in vcb] for s in sentences]
    ts_vcb = set(itertools.chain.from_iterable(ts_sentences))
    print(f"Evaluating {len(ts_sentences)} sentences with vocabulary of {len(ts_vcb)} words")
    # sen_len = [len(s) for s in ts_sentences]
    lkh_vect = model.score(ts_sentences, total_sentences=len(ts_sentences))
    # pb_vect = np.exp(lkh_vect)
    # avoid numeric overload
    # pp_vect = [_perplexity(pb, l) for pb, l in zip(pb_vect, sen_len) if pb != 0 and l > 0]
    # print(f"P(0) for {len([pb for pb in pb_vect if pb == 0])}/{len(ts_sentences)} sentences")
    # pp = [p for p in pp_vect if p < np.inf]
    # return pp_vect, np.mean(pp_vect), np.std(pp_vect)
    pp = _perplexity(lkh_vect, len(ts_vcb))
    return pp


def run_w2v(file_name, raw_file_name, param_r, param_nr, iter):
    print(f"Parameters redundancy: {param_r}\n")
    print("Non-redundant notes training\n")
    nr_models, nr_vcb = train_embeddings(f'{file_name}_train.csv',
                                         param=param_r,
                                         model_name=f'{file_name}_w2v', iter=iter)

    print("Starting non-redundant/redundant notes testing with best NON REDUNDANT model")
    pp_tmp = []
    pp_tmp_raw = []
    for m in nr_models:
        pp_test = evaluate(_read_text(f'{file_name}_test.csv'), m, nr_vcb)
        pp_tmp.append(pp_test)
        pp_test_raw = evaluate(_read_text(f'{raw_file_name}_test.csv'), m, nr_vcb)
        pp_tmp_raw.append(pp_test_raw)
    print(f"Model perplexity for non-redundant test set {np.mean(pp_tmp)} ({np.std(pp_tmp)})\n")
    print(f"Model perplexity for redundant test set {np.mean(pp_tmp_raw)} ({np.std(pp_tmp_raw)})\n")

    print(f"Parameters w/o redundancy: {param_r}\n")
    print("Redundant notes training\n")
    nr_models, r_vcb = train_embeddings(f'{raw_file_name}_train.csv',
                                        param=param_nr,
                                        model_name=f'{raw_file_name}_w2v',
                                        iter=iter)

    print("Starting redundant/non-redundant note testing with best redundant model")
    pp_tmp = []
    pp_tmp_nr = []
    for m in nr_models:
        pp_test = evaluate(_read_text(f'{raw_file_name}_test.csv'), m, r_vcb)
        pp_tmp.append(pp_test)
        pp_test_nr = evaluate(_read_text(f'{file_name}_test.csv'), m, r_vcb)
        pp_tmp_nr.append(pp_test_nr)
    print(f"Model perplexity for redundant test set: {np.mean(pp_tmp)} ({np.std(pp_tmp)})\n")
    print(f"Model perplexity for non-redundant test set: {np.mean(pp_tmp_nr)} ({np.std(pp_tmp_nr)})\n")


"""
Private functions
"""


def _perplexity(lkh_vect, len_sen):
    """
    Compute perplexity as 2^(-1/N)sum(log(P(s1)), ..., log(P(sN))).
    :param pb: probability of a sentence
    :param len_sen: N sentences
    :return: perplexity
    """
    return 2 ** ((-1 / len_sen) * sum(lkh_vect))


def _read_text(file_name):
    with open(os.path.join(utils.data_path, file_name), 'r') as f:
        rd = csv.reader(f)
        sentences = [r for r in rd]
        return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Investigate psychiatric notes redundancy")
    parser.add_argument('-dt', '--dataset',
                        type=str,
                        dest='dataset',
                        help='Select dataset type')
    config = parser.parse_args(sys.argv[1:])

    if config.dataset == 'wn':
        print('*' * 22)
        print("WITHIN-NOTE WORD2VEC")
        print('*' * 22)
        print('\n')
        run_w2v('wn_sentences', 'raw_wn_sentences', utils.w2v_param_wn_r, utils.w2v_param_wn_nr, iter=10)
    elif config.dataset == 'bn':
        print('*' * 22)
        print("BETWEEN-NOTE WORD2VEC")
        print('*' * 22)
        print('\n')
        run_w2v('bn_sentences', 'raw_bn_sentences', utils.w2v_param_bn_r, utils.w2v_param_bn_nr, iter=10)
    else:
        raise ModuleNotFoundError(
            f"Could not find redundancy dataset {config.dataset}. "
            f"Please specify one of the available dataset types: "
            f"'wn' within note redundancy; 'bn' between note redundancy.")
