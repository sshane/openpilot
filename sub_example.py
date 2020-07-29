from cereal.messaging import SubMaster

sm = SubMaster(['thermal'])  # list of services you want to subscribe to, see service_list.yaml

while 1:  # your loop
  sm.update(0)  # run at the top of the loop to update all services, 0 is to not take any extra time waiting
  openpilot_started = sm['thermal'].started
  if openpilot_started:
    pass  # do stuff!
  else:
    pass  # other stuff
