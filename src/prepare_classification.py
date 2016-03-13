import os, sys, time, re, json

from operator import itemgetter

from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

from classification.zb_math_tokenizer import zbMathTokenizer
from classification.arff_json.ArffJsonCorpus import ArffJsonCorpus
from classification.util import group_and_count, save_csr_matrix, load_csr_matrix, build_csr_matrix

start_time = time.time()

corpus_size = 0
def token2index_map(corpus, tokenizer, min_occ = 10):
    global corpus_size
    global_token_counts = {}

    count = 0
    for doc in corpus:
        count += 1
        if(count % 100 == 0):
            sys.stdout.write("\r" + progress_str(count, corpus_size))
            sys.stdout.flush()
            
        doc_str = doc.data[0] + " " + doc.data[1]
        tokens = zbMathTokenizer.doc2tokens(doc_str, tokenizer)

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


def labels2label_vector_entry(doc_labels, target_class):
    if len(target_class) == 2:
        doc_labels = filter(lambda label: len(label) >= 2, doc_labels)
        doc_labels = map(lambda label: label[:2], doc_labels)
        if target_class in doc_labels:
            return 1
        else:
            return 0
    elif len(target_class) == 3:
        doc_labels = filter(lambda label: len(label) >= 3, doc_labels)
        parent_target_class = target_class[:2]
        top_class_labels = map(lambda label: label[:2], doc_labels)
        if parent_target_class in top_class_labels:
            mid_class_labels = map(lambda label: label[:3], doc_labels)
            if target_class in mid_class_labels:
                return 1
            else:
                return 0
        else:
            return -1
    elif len(target_class) == 5:
        doc_labels = filter(lambda label: len(label) == 5, doc_labels)
        parent_target_class = target_class[:3]
        mid_class_labels = map(lambda label: label[:3], doc_labels)
        if parent_target_class in mid_class_labels:
            leaf_class_labels = doc_labels
            if target_class in leaf_class_labels:
                return 1
            else:
                return 0
        else:
            return -1
    else:
        raise ValueError()


def filter_domain_matrix(tdm, labels):
    domain_indexes = []

    for i in range(tdm.shape[0]):
        if labels[i] != -1:
            domain_indexes.append(i)

    domain_matrix = tdm[domain_indexes, :]
    domain_labels = itemgetter(*domain_indexes)(labels)

    return domain_matrix, domain_labels


def norm_label(label):
    if re.match("[0-9][0-9][a-zA-Z-][0-9][0-9]", label):
        return label.lower()
    elif re.match("[0-9][0-9][a-zA-Z-][xX][xX]", label):
        return label.lower()[:3]
    elif re.match("[0-9][0-9]-[xX][xX]", labels):
        return label[:2]
    else:
        raise ValueError(label)


def parent_class(msc_class):
    if len(msc_class) == 2:
        return "root"
    elif len(msc_class) == 3:
        return msc_class[:2]
    elif len(msc_class) == 5:
        return msc_class[:3]
    else:
        raise ValueError("Invalid msc class: " + str(msc_class))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: python prepare_classification.py -o output_path corpus_file"

    output_path = sys.argv[2]
    corpus_filepath = sys.argv[3]

    corpus_size = file_len(corpus_filepath) - 1

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "filter"))
        os.makedirs(os.path.join(output_path, "classifier"))

    corpus = ArffJsonCorpus(corpus_filepath)
    tokenizer = zbMathTokenizer(replace_acronyms=True, stem=True, filter_stopwords=True)

    if not os.path.exists(os.path.join(output_path, "token2index_map.json")):
        print "building token to index map..."
        start_time = time.time()

        t2i = token2index_map(corpus, tokenizer, min_occ=10)
        f = open(os.path.join(output_path, "token2index_map.json"), "w")
        f.write(json.dumps(t2i))
        f.close()
        print "\n"
    else:
        print "load token to index map..."
        t2i = json.load(open(os.path.join(output_path, "token2index_map.json")))
        print ""

    if not os.path.exists(os.path.join(output_path, "corpus.npz")):
        print "building csr matrix..."

        start_time = time.time()

        feature_vectors = []
        count = 0
        for doc in corpus:
            count += 1
            if count % 100 == 0:
                sys.stdout.write("\r" + progress_str(count, corpus_size))
                sys.stdout.flush()
                
            doc_str = doc.data[0] + " " + doc.data[1]
            tokens = zbMathTokenizer.doc2tokens(doc_str, tokenizer)
            feature_vector = group_and_count(tokens)
            feature_vectors.append(feature_vector)
        
        mat = build_csr_matrix(feature_vectors, token2index_map=t2i)
        save_csr_matrix(mat, os.path.join(output_path, "corpus"))

        del feature_vectors
        print "\n"
    else:
        print "load csr_matrix"
        mat = load_csr_matrix(os.path.join(output_path, "corpus.npz"))
        print ""

    # Save Labels
    if not os.path.exists(os.path.join(output_path, "labels")):
        print "building label file"
        corpus = ArffJsonCorpus(corpus_filepath)
        labels_list = (doc.classes for doc in corpus)

        with open(os.path.join(output_path, "labels"), "w") as f:
            for labels in labels_list:
                normed_labels = set(map(lambda label: norm_label(label), labels))
                f.write(",".join(normed_labels) + "\n")
        del labels_list

    labels = read_labels(os.path.join(output_path, "labels"))

    # determine inst counts to evaluate which classes are suitable for training
    inst_count = {'root': len(labels)}
    for doc_labels in labels:
        s = set()
        s.update(doc_labels)
        s.update(map(lambda l: l[:3], doc_labels))
        s.update(map(lambda l: l[:2], doc_labels))

        for l in s:
            if not l in inst_count:
                inst_count[l] = 0

            inst_count[l] = inst_count[l] + 1

    # determine classes of all levels
    top_level_classes = set()
    mid_level_classes = set()
    leaf_classes = set()
    for doc_labels in labels:
        for label in doc_labels:
            if len(label) >= 2:
                top_level_classes.add(label[:2])
            if len(label) >= 3:
                mid_level_classes.add(label[:3])
            if len(label) == 5:
                leaf_classes.add(label)

    # dump existing classes
    classes = {
      'top': sorted(list(top_level_classes)),
      'mid': sorted(list(mid_level_classes)),
      'leaf': sorted(list(leaf_classes))
    }
    f = open(os.path.join(output_path, "classes.json"), "w")
    f.write(json.dumps(classes))
    f.close()

    # create list containing all classes for looping
    all_classes = []
    all_classes.extend(sorted(list(top_level_classes)))
    all_classes.extend(sorted(list(mid_level_classes)))
    all_classes.extend(sorted(list(leaf_classes)))

    evaluation_file = open(os.path.join(output_path, "evaluation.log"), "w")

    count = 1
    for target_class in all_classes:
        clf_path = os.path.join(output_path, "classifier", "svm-" + target_class + ".pkl")
        clf_config_path = os.path.join(output_path, "classifier", "svm-" + target_class + "-conf.json")

        clf_config = {}

        if not os.path.exists(clf_config_path):
            print "preparing msc class " + target_class + " (" + str(count) + " / " + str(len(all_classes)) + ")"

            num_positives = inst_count[target_class]
            #num_positives = sum([1 for x in label_vector if x == 1])
            num_negatives = inst_count[parent_class(target_class)] - num_positives
            #num_negatives = sum([1 for x in label_vector if x == 0])

            clf_config['num_pos'] = num_positives
            clf_config['num_neg'] = num_negatives

            if num_positives >= 250 and num_negatives >= 250:
                label_vector = map(lambda doc_labels: labels2label_vector_entry(doc_labels, target_class), labels)
                
                domain_matrix, domain_label_vector = filter_domain_matrix(mat, label_vector)

                # load or create tfidf filter
                tfidf_path = os.path.join(output_path, "filter", "tfidf-" + parent_class(target_class) + ".pkl")
                if(not os.path.exists(tfidf_path)):
                    tfidf = TfidfTransformer()
                    tfidf.fit(domain_matrix)
                    joblib.dump(tfidf, tfidf_path) 
                else:
                    tfidf = joblib.load(tfidf_path)

                # apply tfidf
                transformed_mat = tfidf.transform(domain_matrix)

                # fit classifier
                clf = svm.LinearSVC()
                clf.fit(transformed_mat, domain_label_vector)

                # dump classifier
                joblib.dump(clf, clf_path) 

                # determine best threshold for classifier
                classified_docs = clf.decision_function(transformed_mat).tolist()
                prec_rec_points, best_point = prec_rec_curve(classified_docs, domain_label_vector)

                # write classifier config
                clf_config["threshold"] = best_point['threshold']

                mean = sum(classified_docs) / len(classified_docs)
                mean_diff = map(lambda x: abs(mean-x), classified_docs)
                std = sum(mean_diff) / len(mean_diff)

                clf_config["mean"] = mean
                clf_config["std"] = std
                clf_config["prec_rec_points"] = prec_rec_points

                # dump evaluation entry
                evaluation_entry = "msc_class: %(msc_class)s, f1: %(f1)0.3f (p: %(prec)0.3f, r: %(rec)0.3f, t: %(thres)f, inst: %(inst)s)" % {
                    "msc_class": target_class, 
                    "f1": best_point['f1'], 
                    "prec": best_point['precision'], 
                    "rec": best_point['recall'], 
                    "thres": best_point['threshold'],
                    "inst": str(num_positives+num_negatives) + " (" + str(num_positives) + "/" + str(num_negatives) + ")"
                }

                evaluation_file.write(evaluation_entry + "\n")
                evaluation_file.flush()
            else:
                print "not enough instances for class " + target_class

            f = open(clf_config_path, "w")
            f.write(json.dumps(clf_config))
            f.close()

            print ""
        count += 1

    evaluation_file.close()