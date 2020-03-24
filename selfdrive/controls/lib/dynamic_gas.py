from selfdrive.car.toyota.values import CAR as CAR_TOYOTA
from selfdrive.car.honda.values import CAR as CAR_HONDA
from common.numpy_fast import clip, interp


class DynamicGas:
  def __init__(self, CP, candidate):
    self.toyota_candidates = [attr for attr in dir(CAR_TOYOTA) if not attr.startswith("__")]
    self.honda_candidates = [attr for attr in dir(CAR_HONDA) if not attr.startswith("__")]

    self.candidate = candidate
    self.CP = CP

  def update(self, v_ego, lead_data, mpc_TR, blinker_status):
    x, y = [], []  # gasMaxBP, gasMaxV
    if self.CP.enableGasInterceptor:
      if self.candidate == CAR_TOYOTA.COROLLA:
        # x = [0.0, 1.4082, 2.8031, 4.2266, 5.3827, 6.1656, 7.2478, 8.2831, 10.2446, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.058, 32.7102, 35.7633]  # todo: average this and below
        # y = [0.221, 0.225, 0.236, 0.253, 0.275, 0.296, 0.338, 0.363, 0.38, 0.389, 0.4, 0.41, 0.421, 0.458, 0.51, 0.561, 0.617]
        x = [0.0, 1.4082, 2.8031, 4.2266, 5.3827, 6.1656, 7.2478, 8.2831, 10.2447, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.0581, 32.7101, 35.7633]
        y = [0.215, 0.219, 0.23, 0.247, 0.271, 0.292, 0.335, 0.361, 0.38, 0.388, 0.396, 0.41, 0.421, 0.46, 0.513, 0.566, 0.624]
      elif self.candidate == CAR_TOYOTA.RAV4:
        x = [0.0, 1.4082, 2.80311, 4.22661, 5.38271, 6.16561, 7.24781, 8.28308, 10.24465, 12.96402, 15.42303, 18.11903, 20.11703, 24.46614, 29.05805, 32.71015, 35.76326]
        y = [0.234, 0.237, 0.246, 0.26, 0.279, 0.297, 0.332, 0.354, 0.368, 0.377, 0.389, 0.399, 0.411, 0.45, 0.504, 0.558, 0.617]
      elif self.candidate in [CAR_HONDA.PILOT_2019, CAR_HONDA.CIVIC]:
        x = [0.0, 1.4082, 2.80311, 4.22661, 5.38271, 6.16561, 7.24781, 8.28308, 10.24465, 12.96402, 15.42303, 18.11903, 20.11703, 24.46614, 29.05805, 32.71015, 35.76326]
        y = [0.234, 0.237, 0.246, 0.26, 0.279, 0.297, 0.332, 0.354, 0.368, 0.377, 0.389, 0.399, 0.411, 0.45, 0.504, 0.558, 0.617]
    # else:
    #   if self.candidate in [CAR_TOYOTA.CAMRY, CAR_TOYOTA.CAMRYH]:
    #     x = [0.]
    #     y = [0.5]

    if not x:
      if self.CP.enableGasInterceptor:
        x, y = self.CP.gasMaxBP, self.CP.gasMaxV  # if no custom gas profile, use stock gas values
      else:  # else if no gas profile and user doesn't have pedal, no dynamic gas
        return interp(v_ego, self.CP.gasMaxBP, self.CP.gasMaxV)

    gas = interp(v_ego, x, y)

    if lead_data['status']:  # if lead
      if v_ego <= 8.9408:  # if under 20 mph
        x = [0.0, 0.24588812499999999, 0.432818589, 0.593044697, 0.730381365, 1.050833588, 1.3965, 1.714627481]  # relative velocity mod
        y = [gas * 0.9901, gas * 0.905, gas * 0.8045, gas * 0.625, gas * 0.431, gas * 0.2083, gas * .0667, 0]
        gas_mod = -interp(lead_data['v_rel'], x, y)

        x = [0.44704, 1.1176, 1.34112]  # lead accel mod
        y = [1.0, 0.75, 0.625]  # maximum we can reduce gas_mod is 40 percent (never increases mod)
        gas_mod *= interp(lead_data['a_lead'], x, y)

        x = [6.1, 9.15, 15.24]  # as lead gets further from car, lessen gas mod/reduction
        y = [1.0, 0.75, 0.0]
        gas_mod *= interp(lead_data['x_lead'], x, y)

        new_gas = gas + gas_mod

        x = [1.78816, 6.0, 8.9408]  # slowly ramp mods down as we approach 20 mph
        y = [new_gas, (new_gas * 0.6 + gas * 0.4), gas]
        gas = interp(v_ego, x, y)
      else:
        x = [-3.12928, -1.78816, -0.89408, 0, 0.33528, 1.78816, 2.68224]  # relative velocity mod
        y = [-gas * 0.2625, -gas * 0.18, -gas * 0.12, 0.0, gas * 0.075, gas * 0.135, gas * 0.19]
        gas_mod = interp(lead_data['v_rel'], x, y)

        current_TR = lead_data['x_lead'] / v_ego
        x = [mpc_TR - 0.22, mpc_TR, mpc_TR + 0.2, mpc_TR + 0.4]
        y = [-gas_mod * 0.15, 0.0, gas_mod * 0.25, gas_mod * 0.4]
        gas_mod -= interp(current_TR, x, y)

        gas += gas_mod

        if blinker_status:
          x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
          y = [1.0, 1.2875, 1.4625]
          gas *= interp(v_ego, x, y)

    return clip(gas, 0.0, 1.0)
