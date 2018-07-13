import chainer
from chainer.training import extentions
import json
import requests

class SlackNortifiler(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
        self,
        model,
        slack_url,
        key,
    ):

        self.model = model
        self.SLACKURL = slack_url
        self.key = key

    def __call__(self, iterator):
        
