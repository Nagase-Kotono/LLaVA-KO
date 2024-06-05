from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, PhiModel, PhiConfig, PhiForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaPhiConfig(PhiConfig):
    """LlavaPhi 모델을 위한 구성 클래스."""
    model_type = "llava_phi"

class LlavaPhiModel(LlavaMetaModel, PhiModel):
    """LlavaMetaModel과 PhiModel을 상속받는 LlavaPhi 모델 클래스."""
    config_class = LlavaPhiConfig

    def __init__(self, config: PhiConfig):
        """주어진 구성으로 LlavaPhiModel을 초기화합니다.

        Args:
            config (PhiConfig): 모델의 구성.
        """
        super(LlavaPhiModel, self).__init__(config)

class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    """LlavaPhi를 위한 Causal Language Model 클래스. PhiForCausalLM과 LlavaMetaForCausalLM을 상속받습니다."""
    config_class = LlavaPhiConfig

    def __init__(self, config):
        """주어진 구성으로 LlavaPhiForCausalLM을 초기화합니다.

        Args:
            config (PhiConfig): 모델의 구성.
        """
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 가중치 초기화 및 최종 처리 적용
        self.post_init()

    def get_model(self):
        """모델 인스턴스를 반환합니다.

        Returns:
            LlavaPhiModel: 기본 모델 인스턴스.
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """모델의 forward 패스를 수행합니다.

        Args:
            input_ids (torch.LongTensor, optional): 입력 ID들.
            attention_mask (torch.Tensor, optional): 어텐션 마스크.
            position_ids (torch.LongTensor, optional): 위치 ID들.
            past_key_values (List[torch.FloatTensor], optional): 과거 키 값들.
            inputs_embeds (torch.FloatTensor, optional): 입력 임베딩.
            labels (torch.LongTensor, optional): 학습을 위한 레이블.
            use_cache (bool, optional): 캐시 사용 여부.
            output_attentions (bool, optional): 어텐션 출력 여부.
            output_hidden_states (bool, optional): 히든 스테이트 출력 여부.
            images (torch.FloatTensor, optional): 이미지 입력.
            image_sizes (List[List[int]], optional): 이미지 크기.
            return_dict (bool, optional): 딕셔너리 반환 여부.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: 모델의 출력.
        """
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """주어진 입력과 이미지로부터 텍스트를 생성합니다.

        Args:
            inputs (torch.Tensor, optional): 입력 텐서.
            images (torch.Tensor, optional): 이미지 입력.
            image_sizes (torch.Tensor, optional): 이미지 크기.
            **kwargs: 추가적인 키워드 인자들.

        Returns:
            Union[GenerateOutput, torch.LongTensor]: 생성된 출력.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds`는 지원되지 않습니다")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """텍스트 생성을 위한 입력을 준비합니다.

        Args:
            input_ids: 입력 ID들.
            past_key_values: 과거 키 값들.
            inputs_embeds: 입력 임베딩.
            **kwargs: 추가적인 키워드 인자들.

        Returns:
            dict: 생성을 위한 준비된 입력.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# 구성과 모델 등록
AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
