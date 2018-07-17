import chainer
from chainer.training import extentions
import json
import requests

class SlackNortifiler(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
        self,
        entries,
        log_report='LogReport',
        slack_url,
    ):
        self._entries = entries
        self._log_report= log_report
        self.SLACKURL = slack_url

        self._log_len= 0

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary

    def _print(self, observation):
        
