# -*- coding: utf-8 -*-
"""Use the spacy library to extract linguistic features.
"""

from warnings import catch_warnings, simplefilter

import spacy


class spacyCleanNLP:
    """A class to call spacy and output normalized tables"""

    def __init__(self, model_name='en'):
        with catch_warnings():
            simplefilter("ignore")
            self.nlp = spacy.load(name=model_name)

    def parseDocument(self, text, id):
        with catch_warnings():
            simplefilter("ignore")
            doc = self.nlp(text)

        sent_index = {}
        sid = 1
        for sent in doc.sents:
            sent_index[sent.start] = sid
            sid += 1

        token = get_token(doc, id)
        ent = get_entity(doc, id, sent_index)

        return {"token": token, "entity": ent}


def get_token(doc, id):
    token = {
        "id": [],
        "sid": [],
        "tid": [],
        "token": [],
        "lemma": [],
        "upos": [],
        "xpos": [],
        "tid_source": [],
        "relation": []
    }

    sid = 1
    for x in doc.sents:
        # spacy counts tokens over the whole doc, but we
        # want it within a sentence as with corenlp. So
        # save the first at sentence start and substract
        # it off
        start_token_i = x[0].i

        # Now, parse the actual tokens, starting at 1
        tid = 1
        for word in x:
            if word.dep_ == "ROOT":
                dep_id = 0
            else:
                dep_id = word.head.i - start_token_i + 1

            this_text = word.text
            this_lemma = word.lemma_
            this_text = this_text.replace("\"", "\\\'")
            this_text = this_text.replace("\'", "\\\'")
            this_lemma = this_lemma.replace("\"", "\\\'")
            this_lemma = this_lemma.replace("\'", "\\\'")

            token['id'].append(id)
            token['sid'].append(sid)
            token['tid'].append(tid)
            token['token'].append(this_text)
            token['lemma'].append(this_lemma)
            token['upos'].append(word.pos_)
            token['xpos'].append(word.tag_)
            token['tid_source'].append(dep_id)
            token['relation'].append(word.dep_)

            tid += 1
        sid += 1

    return token


def get_entity(doc, id, sent_index):
    evals = {
        "id": [],
        "sid": [],
        "tid": [],
        "tid_end": [],
        "entity_type": [],
        "entity": []
    }

    for ent in doc.ents:

        sid = sent_index.get(ent.sent.start, -1)
        tid_start = ent.start - ent.sent.start + 1
        tid_end = ent.end - ent.sent.start
        entity = ent.text.replace('"','')

        evals['id'].append(id)
        evals['sid'].append(sid)
        evals['tid'].append(tid_start)
        evals['tid_end'].append(tid_end)
        evals['entity_type'].append(ent.root.ent_type_)
        evals['entity'].append(entity)

    return evals
