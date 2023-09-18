import os
import argparse
import requests
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config


MAX_SOURCE_LENGTH = 1024


class ReviewerModel(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        factor = self.config.initializer_factor
        self.cls_head.weight.data.normal_(mean=0.0, \
                                          std=factor * ((self.config.d_model) ** -0.5))
        self.cls_head.bias.data.zero_()

    def forward(
            self, *argv, **kwargs
    ):
        r"""
        Doc from Huggingface transformers:
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        if "cls" in kwargs:
            assert (
                    "input_ids" in kwargs and \
                    "labels" in kwargs and \
                    "attention_mask" in kwargs
            )
            return self.cls(
                input_ids=kwargs["input_ids"],
                labels=kwargs["labels"],
                attention_mask=kwargs["attention_mask"],
            )
        if "input_labels" in kwargs:
            assert (
                    "input_ids" in kwargs and \
                    "input_labels" in kwargs and \
                    "decoder_input_ids" in kwargs and \
                    "attention_mask" in kwargs and \
                    "decoder_attention_mask" in kwargs
            ), "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            if "encoder_loss" not in kwargs:
                encoder_loss = True
            else:
                encoder_loss = kwargs["encoder_loss"]
            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask,
                                       decoder_attention_mask, encoder_loss)
        return super().forward(*argv, **kwargs)

    def cls(
            self,
            input_ids,
            labels,
            attention_mask,
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]
        first_hidden = nn.Dropout(0.3)(first_hidden)
        logits = self.cls_head(first_hidden)
        loss_fct = CrossEntropyLoss()
        if labels != None:
            loss = loss_fct(logits, labels)
            return loss
        return logits

    def review_forward(
            self,
            input_ids,
            input_labels,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            encoder_loss=True
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        decoder_inputs = self._shift_right(decoder_input_ids)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:  # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if encoder_loss:
            # print(self.encoder.get_input_embeddings().weight.shape)
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
            # cls_logits = self.cls_head(hidden_states)
        lm_logits = self.lm_head(sequence_output)
        if decoder_input_ids is not None:
            lm_loss_fct = CrossEntropyLoss(ignore_index=0)  # Warning: PAD_ID should be 0
            loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            if encoder_loss and input_labels is not None:
                cls_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            return loss
        return cls_logits, lm_logits


def prepare_models():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")

    tokenizer.special_dict = {
        f"<e{i}>": tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }
    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    config = T5Config.from_pretrained("microsoft/codereviewer")
    model = ReviewerModel.from_pretrained("microsoft/codereviewer", config=config)

    model.eval()
    return tokenizer, model


def pad_assert(tokenizer, source_ids):
    source_ids = source_ids[:MAX_SOURCE_LENGTH - 2]
    source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    pad_len = MAX_SOURCE_LENGTH - len(source_ids)
    source_ids += [tokenizer.pad_id] * pad_len
    assert len(source_ids) == MAX_SOURCE_LENGTH, "Not equal length."
    return source_ids


def encode_diff(tokenizer, diff, msg, source):
    difflines = diff.split("\n")[1:]  # remove start @@
    difflines = [line for line in difflines if len(line.strip()) > 0]
    map_dic = {"-": 0, "+": 1, " ": 2}

    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2

    labels = [f(line[0]) for line in difflines]
    difflines = [line[1:].strip() for line in difflines]
    inputstr = "<s>" + source + "</s>"
    inputstr += "<msg>" + msg
    for label, line in zip(labels, difflines):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<del>" + line
        else:
            inputstr += "<keep>" + line
    source_ids = tokenizer.encode(inputstr, max_length=MAX_SOURCE_LENGTH, truncation=True)[1:-1]
    source_ids = pad_assert(tokenizer, source_ids)
    return source_ids


class FileDiffs(object):
    def __init__(self, diff_string):
        diff_array = diff_string.split("\n")
        self.file_name = diff_array[0]
        self.file_path = self.file_name.split("a/", 1)[1].rsplit("b/", 1)[0]
        self.diffs = []
        for line in diff_array[4:]:
            if line.startswith("@@"):
                self.diffs.append("")
            if self.diffs:
                self.diffs[-1] += "\n" + line

#TODO Conviene agregar los parámetros repository, commit y organization para que la función tenga valores por defecto para pruebas iniciales
def review_commit(repository="NOMBREDELREPOGITHUBAQUI", # Cambio 1: Agregamos el parámetro repository
                 commit='COMMITIDAQUI', # Cambio 2: Agregamos el parámetro commit id
                 github_base_url=os.environ.get("GITHUB_URL"),
                 github_api_base_url=os.environ.get("GITHUB_API_URL"),
                 organization="NOMBREDEORGANIZACIONGITHUBAQUI", # Cambio 3: Agregamos el parámetro organization
                 token=os.environ.get("GITHUB_TOKEN"),
                 verbose=True):  # Agregamos el parámetro verbose

    tokenizer, model = prepare_models()

    # Encabezado de autenticación para las solicitudes de la API de GitHub
    headers = {
        "Authorization": f"token {token}"
    }

    if verbose:
        print("Fetching commit metadata and diff...")

    # Get diff and commit metadata from GitHub API
    commit_metadata = requests.get(f"{github_api_base_url}/repos/{organization}/{repository}/commits/{commit}",
                                   headers=headers).json()
    msg = commit_metadata["commit"]["message"]
    diff_data = requests.get(f"{github_api_base_url}/repos/{organization}/{repository}/commits/{commit}",
                             headers={**headers, "Accept": "application/vnd.github.diff"})
    code_diff = diff_data.text

    # Parse diff into FileDiffs objects
    files_diffs = list()
    for file in code_diff.split("diff --git"):
        if len(file) > 0:
            fd = FileDiffs(file)
            files_diffs.append(fd)

    if verbose:
        print(f"Parsed {len(files_diffs)} file diffs.")

    # Generate comments for each diff
    reviewed_files = []
    total_diffs = sum(len(fd.diffs) for fd in files_diffs)
    completed_diffs = 0

    for fd in files_diffs:
        file_comments = []
        source = requests.get(f"{github_base_url}/{organization}/{repository}/raw/{commit}/{fd.file_path}",
                              headers=headers).text

        for diff in fd.diffs:
            completed_diffs += 1
            if verbose:
                print(f"Analyzing diff {completed_diffs}/{total_diffs} in file {fd.file_path}...")

            inputs = torch.tensor([encode_diff(tokenizer, diff, msg, source)], dtype=torch.long).to("cpu")
            inputs_mask = inputs.ne(tokenizer.pad_id)
            logits = model(
                input_ids=inputs,
                cls=True,
                attention_mask=inputs_mask,
                labels=None,
                use_cache=True,
                num_beams=5,
                early_stopping=True,
                max_length=100
            )
            needs_review = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            if not needs_review:
                continue
            preds = model.generate(inputs,
                                   attention_mask=inputs_mask,
                                   use_cache=True,
                                   num_beams=5,
                                   early_stopping=True,
                                   max_length=100,
                                   num_return_sequences=2
                                   )
            preds = list(preds.cpu().numpy())
            pred_nls = [tokenizer.decode(_id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for _id in preds]
            file_comments.append((diff, pred_nls))
            print(diff)
            for nl in pred_nls:
                print(nl)
            
        reviewed_files.append((fd.file_path, file_comments))

    return reviewed_files  # Cambio 5: Devolvemos la lista de archivos revisados en lugar de la cadena de texto


description = "An interface for running " \
              "\"Microsoft CodeBERT CodeReviewer: Pre-Training for Automating Code Review Activities.\" " \
              "(microsoft/codereviewer) on GitHub commits."
examples = [
    ["p4vv37", "ueflow", "610a8c7b02b946bc9e5e26e6dacbba0e2abba259"],
    ["microsoft", "vscode", "378b0d711f6b82ac59b47fb246906043a6fb995a"],
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review a GitHub commit using CodeBERT CodeReviewer")
    parser.add_argument("user", help="GitHub username")
    parser.add_argument("repository", help="GitHub repository name")
    parser.add_argument("commit", help="Commit ID")

    # En caso de querer usar este script con argumentos de entrada, podemos sustituir el codigo comentado y comentar la linea de abajo de <output = review_commit()>
    #args = parser.parse_args()
    #output = review_commit(args.user, args.repository, args.commit)
    output = review_commit()
    for filename, comments in output:
        print("-------------------------")
        print(f"Archivo: {filename}")
        print("Comentarios:")
        for diff, comment_list in comments:
            print("Contexto (diff):")
            print(diff)
            for comment in comment_list:
                print(f"Comentario generado: {comment}\n")


with open("output.txt", "w") as f:
    for filename, comments in output:
        f.write("-------------------------\n")
        f.write(f"Archivo: {filename}\n")
        f.write("Comentarios:\n")
        for diff, comment_list in comments:
            f.write("Contexto (diff):\n")
            f.write(diff + "\n")
            for comment in comment_list:
                f.write(f"Comentario generado: {comment}\n\n")