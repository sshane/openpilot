import cereal.messaging as messaging

pm = messaging.PubMaster(['dynamicFollowButton', 'dynamicFollowData'])

while 1:
  model_profile = int(input('model profile: '))
  dat = messaging.new_message()
  dat.init('dynamicFollowData')
  dat.dynamicFollowData.mpcTR = 1.8
  dat.dynamicFollowData.profilePred = model_profile
  pm.send('dynamicFollowData', dat)
  print('sent')

  df_button = int(input('df button: '))
  dat = messaging.new_message()
  dat.init('dynamicFollowButton')
  dat.dynamicFollowData.status = df_button
  pm.send('dynamicFollowButton', dat)


  i = input()
  if i == 'exit':
    break
