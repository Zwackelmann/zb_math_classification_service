import json
import sys
from sklearn.externals import joblib
import os
from zb_math_tokenizer import zbMathTokenizer
from util import group_and_count, build_csr_matrix, parent_class

class zbMathClassifier:
    def __init__(self):
        self.tokenizer = zbMathTokenizer(replace_acronyms=True, stem=True, filter_stopwords=True)
        pass

    def initialize(self, input_dir):
        # if classification context is changed, delete variables beforehand, because 
        # the they might occupy quite a lot of memory
        try:
            del self.t2i
            del self.class_hierarchy
            del self.tfidf
            del self.clf
        except AttributeError:
            pass

        # load token2index map
        self.t2i = json.load(open(os.path.join(input_dir, "token2index_map.json")))

        # load class hierarchy
        classes = json.load(open(os.path.join(input_dir, "classes.json")))

        self.class_hierarchy = {}
        for c in classes['top']:
            self.class_hierarchy[c] = {}

        for c in classes['mid']:
            self.class_hierarchy[c[:2]][c] = {}

        for c in classes['leaf']:
            self.class_hierarchy[c[:2]][c[:3]][c] = {}

        # load tfidf filters
        self.tfidf = {'root': joblib.load(os.path.join(input_dir, "filter", "tfidf-root.pkl"))}

        all_classes = []
        all_classes.extend(classes['top'])
        all_classes.extend(classes['mid'])
        for msc_class in all_classes:
            filter_path = os.path.join(input_dir, "filter", "tfidf-" + msc_class + ".pkl")
            if os.path.exists(filter_path):
                self.tfidf[msc_class] = joblib.load(filter_path)

        del all_classes

        # load classifiers
        self.clf = {}
        all_classes = []
        all_classes.extend(classes['top'])
        all_classes.extend(classes['mid'])
        all_classes.extend(classes['leaf'])
        for msc_class in all_classes:
            (classifier, threshold, prec_rec_points) = zbMathClassifier.load_classifier(input_dir, msc_class)
            if classifier is not None and threshold is not None and prec_rec_points is not None:
                self.clf[msc_class] = (classifier, threshold, prec_rec_points)

        del all_classes

    def classify(self, title, abstract):
        doc = title + " " + abstract
        mat = zbMathClassifier.doc2mat(doc, self.tokenizer, self.t2i)

        matches = []
        zbMathClassifier.classify_mat(mat, self.class_hierarchy, self.clf, self.tfidf, matches, 0.5)

        return matches

    @classmethod
    def classify_mat(cls, doc_mat, class_hierarchy, clf, tfidf, matches, smooth):
        for (msc_class, child_hierarchy) in class_hierarchy.items():
            if msc_class in clf:
                (classifier, threshold, prec_rec_points) = clf[msc_class]

                transformed_mat = tfidf[parent_class(msc_class)].transform(doc_mat)
                clf_value = classifier.decision_function(transformed_mat).tolist()[0]

                if clf_value >= threshold:
                    matches.append((msc_class, zbMathClassifier.clf_value2prop(clf_value, prec_rec_points)))
                    zbMathClassifier.classify_mat(doc_mat, child_hierarchy, clf, tfidf, matches, smooth)

    @classmethod
    def doc2mat(cls, doc, tokenizer, token2index_map):
        tokens = zbMathTokenizer.doc2tokens(doc, tokenizer)
        feature_vector = group_and_count(tokens)

        mat = build_csr_matrix([feature_vector], token2index_map=token2index_map)

        return mat

    @classmethod
    def load_classifier(cls, input_dir, msc_class):
        conf = json.load(open(os.path.join(input_dir, "classifier", "svm-" + msc_class + "-conf.json")))

        if 'threshold' in conf and os.path.exists(os.path.join(input_dir, "classifier", "svm-" + msc_class + ".pkl")):
            classifier = joblib.load(os.path.join(input_dir, "classifier", "svm-" + msc_class + ".pkl"))
            threshold = conf['threshold']
            prec_rec_points = conf['prec_rec_points']
            return (classifier, threshold, prec_rec_points)
        else:
            return (None, None, None)

    @classmethod
    def clf_value2prop(cls, clf_value, prec_rec_points):
        i = 0
        while i < len(prec_rec_points) and prec_rec_points[i]['threshold'] > clf_value:
            i += 1

        if i == 0:
            return 1.0
        elif i == len(prec_rec_points):
            return 0.0
        else:
            threshold_before = prec_rec_points[i-1]['threshold']
            threshold_after = prec_rec_points[i]['threshold']

            precision_before = prec_rec_points[i-1]['precision']
            precision_after = prec_rec_points[i]['precision']

            scale = (clf_value-threshold_after) / (threshold_before-threshold_after)

            return precision_after + scale*(precision_before-precision_after)

if __name__ == "__main__":
    input_dir = "zb_math-2014-clf"

    clf = zbMathClassifier()

    clf.initialize(input_dir)

    # ["37E30","37G20","37J10"]
    print clf.classify(
        title = "Reconnection scenarios and the threshold of reconnection in the dynamics of non-twist maps.",
        abstract = """Reconnection is a global bifurcation of the invariant manifolds of two or more distinct hyperbolic 
        orbits of a non-twist area-preserving map of the annulus, having the same rotation number. We show that for a generic 
        perturbation of an integrable non-twist area-preserving map there exist two possible scenarios of reconnection. At the 
        threshold of reconnection the involved hyperbolic orbits are connected.\\par The threshold of reconnection is defined 
        in terms of the action values on the hype rbolic orbits which reconnect, action being a real-valued function constructed 
        from a primitive function of the map."""
    )