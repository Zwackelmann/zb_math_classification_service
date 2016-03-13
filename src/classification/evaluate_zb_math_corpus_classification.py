import random
from operator import itemgetter
from sklearn import svm
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
import json
from sklearn.decomposition import TruncatedSVD
import itertools
import sys
import time
from sklearn.externals import joblib
import os

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from zb_math_tokenizer import zbMathTokenizer

from arff_json.ArffJsonCorpus import ArffJsonCorpus
from string import ascii_letters, digits
from util import group_and_count, save_csr_matrix, load_csr_matrix, build_csr_matrix

start_time = time.time()

def doc2tokens(doc, tokenizer):
    token_objects = tokenizer.tokenize((doc.data[0] + " " + doc.data[1]))

    text_tokens = []
    for token_object in token_objects:
        if type(token_object) is zbMathTokenizer.TokenString:
            text_tokens.append(filter(lambda c: c in ascii_letters+digits, token_object.str).lower())
        elif type(token_object) is zbMathTokenizer.Reference:
            text_tokens.append(token_object.strMapping())
        elif type(token_object) is zbMathTokenizer.Author:
            text_tokens.append(token_object.strMapping())
        elif type(token_object) is zbMathTokenizer.Formula:
            text_tokens.append(token_object.strMapping())
        else:
            pass
    
    return text_tokens

def token2index_map(corpus, tokenizer, min_occ = 10):
    global_token_counts = {}

    count = 0
    for doc in corpus:
        count += 1
        if(count % 100 == 0):
            sys.stdout.write("\r" + progress_str(count, 556297))
            sys.stdout.flush()
            
        tokens = doc2tokens(doc, tokenizer)

        for token in tokens:
            if token not in global_token_counts:
                global_token_counts[token] = 0

            global_token_counts[token] = global_token_counts[token] + 1

    frequent_tokens = map(lambda x: x[0], filter(lambda x: x[1] > min_occ, global_token_counts.items()))
    sorted_tokens = sorted(frequent_tokens)

    return dict(zip(sorted_tokens, range(len(sorted_tokens))))

def progress_str(curr, total):
    global start_time

    time_elapsed = time.time() - start_time
    curr_progress = float(curr)/total

    estemate_remaining = (1.0-curr_progress)*(time_elapsed/curr_progress)
 
    return "\r%(curr)d / %(total)d (%(per)1.3f%%) after: %(time)d minutes (%(est)d minutes left)" % {
        "curr": curr, 
        "total": total, 
        "per": curr_progress*100, 
        "time": time_elapsed / 60,
        "est": estemate_remaining / 60
    }
            

if __name__ == "__main__":
    corpus = ArffJsonCorpus("/media/simon/INTENSO/Projekte/MIRS/ClassificationFramework/data/arffJson/corpus.json")
    tokenizer = zbMathTokenizer(replace_acronyms=True, stem=True, filter_stopwords=True)

    """print("building token to index map...")

    t = token2index_map(corpus, tokenizer, min_occ=10)
    f = open("token2index_map.json", "w")
    f.write(json.dumps(t))
    f.close()"""

    """t2i = json.load(open("token2index_map.json"))

    feature_vectors = []
    count = 0
    for doc in corpus:
        count += 1
        if count % 100 == 0:
            if(count % 100 == 0):
                sys.stdout.write("\r" + progress_str(count, 556297))
                sys.stdout.flush()
            
        tokens = doc2tokens(doc, tokenizer)
        feature_vector = group_and_count(tokens)
        feature_vectors.append(feature_vector)
    
    mat = build_csr_matrix(feature_vectors, token2index_map=t2i)
    save_csr_matrix(mat, "corpus")"""

    if True:
        msc_classes = [
            "00", "01", "03", "05", "06", "08", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "22", "26", "28", "30", "31", "32", "33", "34", "35", "37", "39", "40", "41", "42",
            "43", "44", "45", "46", "47", "49", "51", "52", "53", "54", "55", "57", "58", "60", "62",
            "65", "68", "70", "74", "76", "78", "80", "81", "82", "83", "85", "86", "90", "91", "92",
            "93", "94", "97"
        ]

        def split_test_train_set(tdm, labels):
            test_indexes = []
            train_indexes = []
            random.seed(0)

            for i in range(tdm.shape[0]):
                if random.random() > 0.2:
                    train_indexes.append(i)
                else:
                    test_indexes.append(i)

            train_matrix = tdm[train_indexes, :]
            test_matrix = tdm[test_indexes, :]
            train_labels = itemgetter(*train_indexes)(labels)
            test_labels = itemgetter(*test_indexes)(labels)

            return train_matrix, train_labels, test_matrix, test_labels

        def read_labels(file):
            labels_list = []
            with open(file) as f:
                for line in f:
                    labels_list.append([x.strip() for x in line.split(",")])

            return labels_list

        def prec_rec_curve(prediction_values, true_labels):
            pred_and_labels = sorted(zip(prediction_values, true_labels), key=lambda x: x[0], reverse=True)
            points = []

            true_positives = 0
            false_positives = 0

            false_negatives = sum(map(lambda x: x[1], pred_and_labels))
            true_negatives = len(pred_and_labels) - false_negatives

            for i in range(1, len(pred_and_labels)):
                if pred_and_labels[i][1] == 1:
                    true_positives += 1
                    false_negatives -= 1
                elif pred_and_labels[i][1] == 0:
                    false_positives += 1
                    true_negatives -= 1
                else:
                    raise ValueError()

                precision = None
                recall = None
                f1 = None

                if true_positives + false_positives > 0:
                    precision = float(true_positives) / float(true_positives + false_positives)

                if true_positives + false_negatives > 0:
                    recall = float(true_positives) / float(true_positives + false_negatives)

                if precision is not None and recall is not None and precision + recall > 0:
                    f1 = 2*(precision*recall)/(precision+recall)

                points.append({"label": pred_and_labels[i][1],
                               "threshold": pred_and_labels[i][0],
                               "true-positives": true_positives,
                               "true-negatives": true_negatives,
                               "false-positives": false_positives,
                               "false-negatives": false_negatives,
                               "precision": precision,
                               "recall": recall,
                               "f1": f1})

            filtered_points = []
            for recall in range(100):
                p = min(points, key=lambda p: abs(p['recall']-(float(recall)/100)))
                filtered_points.append(p)

            best_point = max(points, key=lambda p: p['f1'])

            return filtered_points, best_point


        # Save CSR-Matrix
        """c = corpus.toCsrMatrix()
        save_csr_matrix(c, "abschlussbericht-csr")"""

        # Save Labels
        """corpus = ArffJsonCorpus("../zb_math_cluster_experiments/raw_data/abschlussbericht-corpus.json")
        labels_list = (doc.classes for doc in corpus)

        with open("abschlussbericht-labels", "w") as f:
            for labels in labels_list:
                top_class_labels = set(map(lambda x: x[:2], labels))
                f.write(",".join(top_class_labels) + "\n")"""


        tdm = load_csr_matrix("corpus.npz")
        labels = read_labels("abschlussbericht-labels")

        mats = {}
        def get_transformed_mat(mat, transform_id, transformer_list, test_train):
            global mats
            if mats.get((transform_id, test_train)) is None:
                if transformer_list is not None or not len(transformer_list) == 0:
                    mat_copy = mat
                    for transformer in transformer_list:
                        mat_copy = transformer.transform(mat_copy)

                    mats[(transform_id, test_train)] = mat_copy
                else:
                    mats[(transform_id, test_train)] = mat

            return mats[(transform_id, test_train)]

        train_mat, train_labels, test_mat, test_labels = split_test_train_set(tdm, labels)

        tfidf = TfidfTransformer()
        tfidf.fit(train_mat)

        if not os.path.exists("lsi-250-model.pkl"):
            print "train lsi-250 model"
            lsi250 = TruncatedSVD(n_components=250)
            lsi250.fit(train_mat)
            print "dump lsi-250 model"
            joblib.dump(lsi250, 'lsi-250-model.pkl') 
        else:
            print "load lsi-250 model"
            lsi250 = joblib.load('lsi-250-model.pkl') 

        if not os.path.exists("lsi-500-model.pkl"):
            print "train lsi-500 model"
            lsi500 = TruncatedSVD(n_components=500)
            lsi500.fit(train_mat)
            print "dump lsi-500 model"
            joblib.dump(lsi500, 'lsi-500-model.pkl') 
        else:
            print "load lsi-500 model"
            lsi500 = joblib.load('lsi-500-model.pkl') 

        if not os.path.exists("tfidf-lsi250-model.pkl"):
            print "tfidf-lsi250 model"
            tfidf_lsi250 = TruncatedSVD(n_components=250)
            tfidf_lsi250.fit(tfidf.transform(train_mat))
            print "dump tfidf-lsi-250 model"
            joblib.dump(tfidf_lsi250, 'tfidf-lsi250-model.pkl') 
        else:
            print "load tfidf-lsi250 model"
            tfidf_lsi250 = joblib.load('tfidf-lsi250-model.pkl') 

        if not os.path.exists("tfidf-lsi500-model.pkl"):
            print "tfidf-lsi500 model"
            tfidf_lsi500 = TruncatedSVD(n_components=500)
            tfidf_lsi500.fit(tfidf.transform(train_mat))
            print "dump tfidf-lsi-500 model"
            joblib.dump(tfidf_lsi500, 'tfidf-lsi500-model.pkl') 
        else:
            print "load tfidf-lsi500 model"
            tfidf_lsi500 = joblib.load('tfidf-lsi500-model.pkl') 

        normalizer = Normalizer()

        log = open("classify-log", "w")
        accu = {}
        for target_class in msc_classes:
            train_label_vector = [(1 if target_class in labels else 0) for labels in train_labels]
            test_label_vector = [(1 if target_class in labels else 0) for labels in test_labels]

            for run in [
                        [("raw", None)],
                        [("tfidf", tfidf)],
                        [("tfidf", tfidf), ("norm", normalizer)],
                        [("lsi500", lsi500)], 
                        [("lsi500", lsi500), ("norm", normalizer)],
                        [("tfidf-lsi250", tfidf_lsi250)],
                        [("tfidf-lsi250", tfidf_lsi250), ("norm", normalizer)],
                        [("tfidf-lsi500", tfidf_lsi500)],
                        [("tfidf-lsi500", tfidf_lsi500), ("norm", normalizer)]
                ]:

                (run_desc, transformers) = zip(*run)
                transformers = list(filter(lambda t: t is not None, transformers))
                run_desc = tuple(run_desc)

                transformed_train_mat = get_transformed_mat(train_mat, run_desc, transformers, "train")
                transformed_test_mat = get_transformed_mat(test_mat, run_desc, transformers, "test")

                clf = svm.LinearSVC()
                clf.fit(transformed_train_mat, train_label_vector)

                prec_rec_points, best_point = prec_rec_curve(clf.decision_function(transformed_test_mat).tolist(), test_label_vector)
                print "class: %s, run: %s, f1: %0.3f (p: %0.3f, r: %0.3f)" % (target_class, repr(run_desc), best_point['f1'], best_point['precision'], best_point['recall'])
                log.write(json.dumps({"target-class": target_class, "run": repr(run_desc), "f1": best_point['f1'], "precision": best_point['precision'], "recall": best_point['recall']}) + "\n")
                log.flush()

                run_accu = accu.get(run_desc, [])
                run_accu.append(best_point['f1'])
                accu[run_desc] = run_accu

        for (run_desc, results) in accu.items():
            print str(run_desc) + ": " + str(sum(results) / len(results))

        log.close()
