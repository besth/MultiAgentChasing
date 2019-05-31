from dm_control import mjcf

class ChangePositionsInXML:
    def __init__(self, model_name):
        self.model_name = model_name

    # new_pos = [[prey_x, prey_y, prey_z], [predator_x, predator_y, predator_z]]
    def __call__(self, new_pos):
        filename = 'xmls/' + self.model_name + '.xml'
        mjcf_model = mjcf.from_path(filename)

        prey_body = mjcf_model.worldbody.body['prey']
        predator_body = mjcf_model.worldbody.body['predator']

        new_prey_pos = new_pos[0]
        new_predator_pos = new_pos[1]

        prey_body.pos = new_prey_pos
        predator_body.pos = new_predator_pos
