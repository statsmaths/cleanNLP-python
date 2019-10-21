# -*- coding: utf-8 -*-
"""Use the coreNLP library to extract linguistic features.
"""

from warnings import catch_warnings, simplefilter
import os, sys

import stanfordnlp

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class corenlpCleanNLP:
    """A class to call spacy and output normalized tables"""

    def __init__(self, lang='en'):
        with HiddenPrints():
            with catch_warnings():
                simplefilter("ignore")
                self.nlp = stanfordnlp.Pipeline(lang=lang)

    def parseDocument(self, text, id):
        with catch_warnings():
            simplefilter("ignore")
            doc = self.nlp(text)

        token = get_token(doc, id)

        return {"token": token}


def get_token(doc, id):
    token = {
        "id": [],
        "sid": [],
        "tid": [],
        "token": [],
        "lemma": [],
        "upos": [],
        "xpos": [],
        "feats": [],
        "tid_source": [],
        "relation": []
    }

    sid = 1
    for x in doc.sentences:

        # Now, parse the actual tokens, starting at 1
        tid = 1
        for word in x.words:
            this_text = word.text
            this_lemma = word.lemma
            this_text = this_text.replace("\"", "\\\'")
            this_text = this_text.replace("\'", "\\\'")
            this_lemma = this_lemma.replace("\"", "\\\'")
            this_lemma = this_lemma.replace("\'", "\\\'")

            token['id'].append(id)
            token['sid'].append(sid)
            token['tid'].append(tid)
            token['token'].append(this_text)
            token['lemma'].append(this_lemma)
            token['upos'].append(word.upos)
            token['xpos'].append(word.pos)
            token['feats'].append(word.feats)
            token['tid_source'].append(word.governor)
            token['relation'].append(word.dependency_relation)

            tid += 1

        sid += 1

    return token
