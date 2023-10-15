from typing import Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.esm.modeling_esm import (
    EsmPreTrainedModel,
    EsmModel,
    EsmClassificationHead,
    SequenceClassifierOutput,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x


class EsmClassificationMixin:
    def _classification_loss(self, outputs, logits, labels, return_dict):
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EsmPooledClassification(EsmPreTrainedModel, EsmClassificationMixin):
    def __init__(self, config, num_layers: int = 2):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config, add_pooling_layer=True)
        self.classifier = MLP(config.hidden_size, config.hidden_size, config.num_labels, num_layers)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        return self._classification_loss(outputs, logits, labels, return_dict)


class EsmForMutationClassification(EsmPreTrainedModel, EsmClassificationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            wildtype_input_ids: Optional[torch.LongTensor] = None,
            wildtype_attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            wildtype_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if wildtype_input_ids is not None or wildtype_embeds is not None:
            wildtype_outputs = self.esm(
                input_ids=wildtype_input_ids,
                attention_mask=wildtype_attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=wildtype_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            output = outputs[0] - wildtype_outputs[0]
        else:
            output = outputs[0]

        logits = self.classifier(output)
        return self._classification_loss(outputs, logits, labels, return_dict)
