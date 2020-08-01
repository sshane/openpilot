from common.op_params import opParams
op_params = opParams()

cam = op_params.fork_params['global_df_mod']
print('default: {}'.format(cam.default))
print('allowed types: {}'.format(cam.allowed_types))
print('description: {}'.format(cam.description))
print('has_allowed_types: {}'.format(cam.has_allowed_types))
print('has_description: {}'.format(cam.has_description))
print('hidden: {}'.format(cam.hidden))
print('live: {}'.format(cam.live))
