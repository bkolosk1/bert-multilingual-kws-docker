import os
import json

from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    redirect,
    url_for
)

from flask_restx import Api, Resource, fields, abort, reqparse
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from . import api_functions
from . import keyword_extraction_main as kw
#from .bert_crossling_prep import get_batch, Corpus, batchify, batchify_docs, get_batch_docs, file_to_df
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from transformers import BertTokenizer

from lemmagen3 import Lemmatizer

lemmatizer = Lemmatizer('sl').lemmatize


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0',
          title='API services',
          description='Multilingual keyword extraction REST API')
ns = api.namespace('rest_api', description='REST services API')

args = {
    'max_length': 256,
    'cuda': False,
    'kw_cut': 10,
    'stemmer': lemmatizer,
    'split_docs': True,   
    'bpe': True, 
    'max_vocab_size' : 0,
    'classification': True,
    'adaptive': False,
    'transfer_learning': True,
    'POS_tags': False,
    'bpe': True,
    'masked_lm': False,
    'rnn': False,
    'crf': False,
    'dev_id' : 0,
    'n_ctx' : 256,
    'lang' : 'en',
    'dict_path' : "dict_russian_latvian_estonian_slovenian_croatian_english_bpe_nopos_nornn_nocrf.ptb",
}



kw_model_path = "model_russian_latvian_estonian_slovenian_croatian_english_folder_russian_latvian_estonian_slovenian_croatian_english_loss_0.06955235407170482_epoch_9.pt"
kw_dictionary_path = "dict_russian_latvian_estonian_slovenian_croatian_english_bpe_nopos_nornn_nocrf.ptb"
kw_sp = BertTokenizer.from_pretrained('bert-base-multilingual-uncased',return_dict=False)

kw_model = kw.loadModel(os.path.join("project", "trained_classification_models", kw_model_path), args['cuda'])
kw_dictionary = kw.loadDict(os.path.join("project", "dictionaries", kw_dictionary_path)) 
#kw_sp = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
#kw_sp.Load(os.path.join("project","bpe", "SloBPE.model"))


# input and output definitions
kw_extractor_input = api.model('KeywordExtractorInput', {
    'text': fields.String(required=True, description='Title + lead + body of the article'),
})

kw_extractor_output = api.model('KeywordExtractorOutput', {
    'keywords': fields.List(fields.String, description='Extracted keywords'),
})


@ns.route('/extract_keywords/')
class KeywordExtractor(Resource):
    @ns.doc('Extracts keywords from news article')
    @ns.expect(kw_extractor_input, validate=True)
    @ns.marshal_with(kw_extractor_output)
    def post(self):
        kw_lem = api_functions.extract_keywords(api.payload['text'], kw_model, kw_dictionary, kw_sp, lemmatizer, args)
        return {"keywords": kw_lem}



@ns.route('/health/')
class Health(Resource):
    @ns.response(200, "successfully fetched health details")
    def get(self):
        return {"status": "running", "message": "Health check successful"}, 200, {}
