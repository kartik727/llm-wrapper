from pathlib import Path
import base64
import io
from PIL import Image
from dataclasses import dataclass

from llm_wrappers.io_objects import BaseMessage
from llm_wrappers.io_objects.openai_io_object import Role

class ImageObject:
    def __init__(self, *, image:bytes=None, image_type:str=None,
            image_url:str=None):
        assert (image is not None) ^ (image_url is not None), \
            'Exactly one of `image` or `image_url` must be specified'
        
        if image is None:
            self._image : str = None
            self._image_type : str = None
            self._image_url : str = image_url
        else:
            self._image : str = self.encode_image(image)
            self._image_type : str = self.validate_image_type(image_type)
            self._image_url : str = None

    @property
    def content(self):
        if self._image is not None:
            url = f'data:image/{self._image_type};base64,{self._image}'
        else:
            url = self._image_url

        return {
            'type': 'image_url',
            'image_url': {
                'url': url
            }
        }
    
    @staticmethod
    def encode_image(image: bytes)->str:
        return base64.b64encode(image).decode('utf-8')
    
    @staticmethod
    def validate_image_type(image_type:str)->str:
        if image_type == 'png':
            return 'png'
        elif image_type in ('jpg', 'jpeg'):
            return 'jpeg'
        else:
            raise ValueError(f'Invalid image type: {image_type}')
    
    @classmethod
    def from_file(cls, file_path: str|Path):
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            image = f.read()
        image_type = file_path.suffix[1:]
        return cls(image=image, image_type=image_type)
    
    @classmethod
    def from_PIL_image(cls, image: Image.Image):
        with io.BytesIO() as f:
            image.save(f, format='PNG')
            image = f.getvalue()
        return cls(image=image, image_type='png')

@dataclass
class OpenAIVisionMessage(BaseMessage):
    role: Role
    text_prompt: str
    images: list[ImageObject]

    @property
    def formatted_msg(self)->dict:
        if len(self.images) == 0:
            content = self.text_prompt
        else:
            text_content = {
                'type': 'text',
                'text': self.text_prompt
            }
            image_content = [image.content for image in self.images]
            content = [text_content] + image_content

        return {'role' : self.role.value, 'content' : content}
