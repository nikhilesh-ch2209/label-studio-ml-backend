import pickle
import os
import numpy as np
import requests
import json
from uuid import uuid4
import os
import re
import json
import pickle
import shutil
import tempfile
import pandas as pd

from collections import Counter
from spacy.lang.en import English

import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report
from urllib.parse import urlparse

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_env


HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY')

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')


class SimpleTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        print("Parsed Lebel Config: ", self.parsed_label_config)
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.nlp = English()
        self.training_args = {   "min_child_weight": 1,
                            "learning_rate": 0.4,
                            "subsample":0.8,
                            "eta":0.1,
                            "eval_metric":['merror','mlogloss'],
                            "objective":'multi:softmax',
                            "early_stopping_rounds":10,
                            "max_depth":5,
                            "num_class": 2
                                  }

        mapping = json.load(open("category_config_map.json", "r"))

        if not self.train_output:
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            self.reset_model(self.training_args)
            # This is an array of <Choice> labels
            self.labels = self.info['labels']
            # make some dummy initialization
            # self.model.fit(X=self.labels, y=list(range(len(self.labels))))
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            with open(self.model_file, mode='rb') as f:
                self.model = pickle.load(f)
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))

    def remove_stopwords(self,wordlist):
        i=0
        while i<len(wordlist):
            wordlist[i] = wordlist[i].lower()
            lexeme = self.nlp.vocab[wordlist[i]]
            if lexeme.is_stop:
                wordlist.pop(i)
                i-=1
            i+=1
        return wordlist

    def mask_num(self,x):
        #res = re.sub(r'\w*\d\w*', "<num>", x)
        res = re.sub(r'[0-9]+', "<num>", x)
        return res

    def clean_text(self,x):
        x = x.replace("/"," ")
        x = x.replace("@"," ")
        x = x.replace("|"," ")
        x = x.replace("-"," ")
        #x = x.replace("."," ")
        #x = x.replace(";"," ")
        x = re.sub(" +", " ", x)
        x = re.sub("\(", "", x)
        x = re.sub("\)", "", x)
        x = re.sub(" +", " ", x)
        wordlist = x.split(" ")
        x = " ".join(self.remove_stopwords(wordlist))
        x = self.mask_num(x)
        return x

    def token_freq(self,df):
        df_temp = pd.DataFrame()
        df_temp["voucher_tokens"] = df["processed_voucher_details"].apply(lambda x: x.split())
        counter = Counter()

        df_temp['voucher_tokens'].map(counter.update)
        return counter

    def pick_keyword_containing_datapoints(self,df,keywords):
        df_filtered = pd.DataFrame()
        for i,voucher in enumerate(df["processed_voucher_details"].values.tolist()):
            for word in keywords:
                if word in voucher:
                    df_filtered = df_filtered.append(df.iloc[i])
                    break

        return df_filtered

    def data_split_train_test(self,X,y,X_train,y_train,X_val,y_val):
        X_train_subset, y_train_subset  = X[:int(0.9*len(X))], y[:int(0.9*len(X))]
        X_val_subset, y_val_subset = X[int(0.9*len(X))+1:], y[int(0.9*len(X))+1:]

        X_train = pd.concat([X_train,X_train_subset])
        y_train = pd.concat([y_train,y_train_subset])
        X_val = pd.concat([X_val,X_val_subset])
        y_val = pd.concat([y_val,y_val_subset])

        return X_train, y_train, X_val, y_val


    def data_filter_pipeline_autocategorized(self, df, nlp, mapping):
        categories = df["Category "].unique()
        df_filtered_final = pd.DataFrame()

        X_train = pd.Series()
        X_test = pd.Series()
        y_train = pd.Series()
        y_test = pd.Series()
        X_val = pd.Series()
        y_val = pd.Series()

        for cat in categories:
            df_subset = df[df["Category "]==cat]

            df_subset = df_subset.dropna(subset=["Voucher Details","Category "])

            # #category mapping
            # df_subset["category_ours"] = df_subset["TRANSACTION CATEGORY"].apply(lambda x: mapping[x] if x in mapping else "UNDEFINED")
            # df_subset = df_subset[df_subset["category_ours"]!="UNDEFINED"]


            #category mapping
            df_subset["category_ours"] = df_subset["Category "].apply(lambda x: mapping[x])
            df_subset["category_ours_processed"] = df_subset["category_ours"].apply(lambda x: x.split("_",1)[1])
            df_subset["processed_voucher_details"] = df_subset["Voucher Details"].apply(lambda x: self.clean_text(x))

            keywords_count = self.token_freq(df_subset).most_common(20)
            keywords = [tup[0] for tup in keywords_count]

            #keywords manipulation
            try:
                keywords.remove("<num>")
            except ValueError:
                pass

            df_filtered = self.pick_keyword_containing_datapoints(df_subset,keywords)

            df_filtered = df_filtered.sample(frac=1)
            X = df_filtered.processed_voucher_details
            y = df_filtered.category_ours_processed

            if len(X) + len(y) >= 10:
                X_train , X_test , y_train , y_test, X_val, y_val = self.data_split_train_test(X,y,X_train,X_test,X_val,y_val)
            else:
                pass
            df_filtered_final = pd.concat([df_filtered_final,df_filtered])


        return df_filtered_final,X_train.values.tolist(),X_test.values.tolist(),y_train.values.tolist(),y_test.values.tolist(),X_val.values.tolist(),y_val.values.tolist()



    def reset_model(self, training_args):
        xgb_classifier = xgb.XGBClassifier(min_child_weight=training_args["min_child_weight"],
                                        learning_rate=training_args["learning_rate"],
                                        subsample=training_args["subsample"],
                                        eta=training_args["eta"],
                                        eval_metric=['merror','mlogloss'],
                                        objective=training_args["objective"],
                                        # early_stopping_rounds=training_args["early_stopping_rounds"],
                                        max_depth=training_args["max_depth"],
                                        num_class=training_args["num_class"]
                                        )


        self.model = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', xgb_classifier),
                        ])

    def predict(self, tasks, **kwargs):
        # collect input texts
        input_texts = []
        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

        # get model predictions
        probabilities = self.model.predict_proba(input_texts)
        print('=== probabilities >', probabilities)
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': str(score)})
        print("PREDICTIONS: ", predictions)

        return predictions

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)

    def fit(self, annotations, workdir=None, **kwargs):
        # check if training is from web hook
        print("STARTED TRAINING")
        if kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            tasks = self._get_annotated_dataset(project_id)
        # ML training without web hook
        else:
            tasks = annotations

        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for task in tasks:
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0]
            # get input text from task data
            if annotation.get('skipped') or annotation.get('was_cancelled'):
                continue

            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

            # get an annotation
            output_label = annotation['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        print(f'Start training on {len(input_texts)} samples')

        self.reset_model(self.training_args)
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=output_labels_idx)
        print("INPUT_TEXTS: ", input_texts[:10])
        print("OUTPUT LABELS: ", output_labels[:10])

        self.model.fit(input_texts, output_labels_idx, **{"clf__sample_weight":sample_weights})

        # save output resources
        workdir = workdir or os.getenv('MODEL_DIR')
        model_name = str(uuid4())[:8]
        if workdir:
            model_file = os.path.join(workdir, f'{model_name}.pkl')
        else:
            model_file = f'{model_name}.pkl'
        print(f'Save model to {model_file}')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output

    def train_model(self,X_train,y_train, **training_args):
        print("Training model")
        print(training_args)

        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train)

        # xgb_classifier = xgb.XGBClassifier(min_child_weight=training_args["min_child_weight"],
        #                                 learning_rate=training_args["learning_rate"],
        #                                 subsample=training_args["subsample"],
        #                                 eta=training_args["eta"],
        #                                 eval_metric=['merror','mlogloss'],
        #                                 objective=training_args["objective"],
        #                                 early_stopping_rounds=training_args["early_stopping_rounds"],
        #                                 #num_class=53,
        #                                 max_depth=training_args["max_depth"],
        #                                 )


        # model = Pipeline([('vect', CountVectorizer()),
        #                 ('tfidf', TfidfTransformer()),
        #                 ('clf', xgb_classifier),
        #                 ])
        self.model.fit(X_train, y_train, **{"clf__sample_weight":sample_weights})
        return self.model



    def run_pipeline(self, csv, mapping, model_data, schema, org_name, **training_args):
        """Runs the entire training pipeline and saves the trained model in specified directory

        Args:
            csv(str): Path to the training data
            schema(dict): dictionary of mappings of standard column names to names in the training data. must contain DESCRIPTION and TRANSACTION CATEGORY keys
            mapping(dict): dictionary of mappings of client data categories to our defined categories
            save_model_dir_path(str): path to where model will be saved, should be in an existing directory

        Returns:
            model(object): returns an object of the trained model

            Also saves model at save_model_path
        """

        df_filtered_final, X_train, y_train,X_val,y_val = self.data_filter_pipeline_autocategorized(csv,schema,mapping)

        print("Training model...")
        model = self.train_model(X_train,y_train, **training_args)

        y_pred_val = model.predict(X_val)

        validation_acc = accuracy_score(y_pred_val, y_val)
        result = {}

        result["validation_acc"] = validation_acc * 100
        result["classification_report"] = classification_report(y_val,y_pred_val)
        result["model_state"] = str(model.get_params())

        model_data.state = 'SUCCESS'
        model_data.meta_data['result'] = result
        model_data.save(using=org_name)
        return model, result

