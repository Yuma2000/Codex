import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TimePM-v0',
    entry_point='Reinf_Envs.envs:TimePM',
)