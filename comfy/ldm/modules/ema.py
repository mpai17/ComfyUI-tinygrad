from tinygrad import Tensor


class LitEma:
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.decay = Tensor([decay])
        self.num_updates = Tensor([0]) if use_num_upates else Tensor([-1])

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                setattr(self, s_name, p.detach())

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.num_updates = Tensor([0])

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        # No gradient context needed in tinygrad
        m_param = dict(model.named_parameters()) if hasattr(model, 'named_parameters') else {}
        shadow_params = {name: getattr(self, name) for name in self.m_name2s_name.values() if hasattr(self, name)}

        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                if sname in shadow_params:
                    shadow_params[sname] = shadow_params[sname] - one_minus_decay * (shadow_params[sname] - m_param[key])
                    setattr(self, sname, shadow_params[sname])
            else:
                assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters()) if hasattr(model, 'named_parameters') else {}
        shadow_params = {name: getattr(self, name) for name in self.m_name2s_name.values() if hasattr(self, name)}
        for key in m_param:
            if m_param[key].requires_grad:
                if hasattr(m_param[key], 'assign'):
                    m_param[key].assign(shadow_params[self.m_name2s_name[key]])
                else:
                    m_param[key] = shadow_params[self.m_name2s_name[key]]
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.detach() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if hasattr(param, 'assign'):
                param.assign(c_param)
            else:
                param = c_param
