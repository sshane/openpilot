from common.op_params import opParams
from selfdrive.controls.lib.dynamic_follow.df_manager import dfManager

op_params = opParams()
df_manager = dfManager(op_params)

while 1:
  df_out = df_manager.update()
  print('user_profile: {}'.format(df_out.user_profile))
  print('user_profile_text: {}'.format(df_out.user_profile_text))
  print('model_profile: {}'.format(df_out.model_profile))
  print('model_profile_text: {}'.format(df_out.model_profile_text))
  print('changed: {}'.format(df_out.changed))
  print('is_auto: {}'.format(df_out.is_auto))
  print('---')

  i = input()
  if i == 'exit':
    break
