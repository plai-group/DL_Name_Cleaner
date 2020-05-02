import torch
import string
from Generators.NameGenerator import NameGenerator
from Constant import DEVICE, TITLES, SUFFIXES

TITLE_PROBS = torch.FloatTensor([1/len(TITLES)] * len(TITLES)).to(DEVICE)
SUFFIX_PROBS = torch.FloatTensor([1/len(SUFFIXES)] * len(SUFFIXES)).to(DEVICE)


class FullNameGenerator():
    def __init__(self, first_config_pth: str = 'Config/Pretrained/first.json', last_config_pth: str = 'Config/Pretrained/last.json', first_wght_pth: str = 'Weights/Pretrained/first.path.tar', last_wght_pth: str = 'Weights/Pretrained/last.path.tar'):
        super(FullNameGenerator, self).__init__()
        self.fn_generator = NameGenerator(
            first_config_pth, first_wght_pth)
        self.ln_generator = NameGenerator(
            last_config_pth, last_wght_pth)

    def sampleAllNameComponents(self, has_middle: bool, has_title: bool, has_suffix: bool):
        first = self.fn_generator.generateName(self.sampleFirstLength())
        last = self.ln_generator.generateName(self.sampleLastLength())

        middles = []
        if has_middle:
            full_middle = ''
            middle_classification = ''
            num_middles = self.sampleNumMiddleNames()

            for i in range(num_middles):
                middles.append(self.fn_generator.generateName(
                    self.sampleUniformLength()))

        title = None
        if has_title:
            title_idx = int(torch.distributions.Categorical(
                TITLE_PROBS).sample().item())
            title = TITLES[title_idx]

        suffix = None
        if has_suffix:
            suffix_idx = int(torch.distributions.Categorical(
                SUFFIX_PROBS).sample().item())
            suffix = SUFFIXES[suffix_idx]

        return first, middles, last, title, suffix

    def sampleUniformLength(self):
        return int(torch.distributions.Categorical(torch.FloatTensor([1/12] * 12)).sample().item()) + 1

    def sampleFirstLength(self):
        # This probability dictionary was generated from FB name data
        probs_dict = {7: 0.14771852018788723, 4: 0.1875379024407351, 5: 0.2721584477622317, 6: 0.21436671567742333, 8: 0.06421742726838446, 3: 0.06738360597444391, 9: 0.027237021268569788, 11: 0.0035402552149823583, 2: 0.008484761007749555, 10: 0.0058731792484854155, 12: 0.0007644787012189057, 14: 0.00015739482892918714, 1: 2.8584600914932915e-05, 13: 0.0003414251421766981, 15: 8.257366513980493e-05, 16: 4.4815026710005054e-05,
                      18: 1.4010191476378298e-05, 17: 2.2429495873009397e-05, 20: 5.517612019724298e-06, 19: 8.294736794592172e-06, 21: 3.597805447124343e-06, 23: 1.9857541266205642e-06, 24: 1.3922261404350818e-06, 22: 2.447387004764828e-06, 30: 2.2715268607098702e-07, 26: 7.400781062312803e-07, 27: 5.568904561740327e-07, 28: 3.8103031211907503e-07, 25: 1.0258508403205867e-06, 29: 2.7844522808701634e-07, 40: 7.327506002289904e-09}

        return self.sampleFromProbsDict(probs_dict)

    def sampleLastLength(self):
        # This probability dictionary was generated from FB name data
        probs_dict = {5: 0.2100207947101849, 7: 0.17996100621251723, 8: 0.11553042484096908, 3: 0.03997644478281118, 4: 0.1138967946111999, 6: 0.2318648605821442, 9: 0.058263049060305265, 10: 0.0250731935196788, 2: 0.00979029313165891, 11: 0.009392976761946158, 1: 0.0013191820114049327, 12: 0.003217669652894908,
                      13: 0.0010961859867165311, 15: 0.0001324438319261756, 14: 0.00039466930820212127, 16: 4.668660201463465e-05, 18: 7.2378224733525614e-06, 17: 1.3401694276677887e-05, 20: 9.453791109394672e-07, 19: 1.4899174788406004e-06, 21: 1.0588246042522033e-07, 22: 9.075639465018885e-08, 27: 5.2941230212610165e-08}

        return self.sampleFromProbsDict(probs_dict)

    def sampleFromProbsDict(self, probs_dict: dict):
        probs_tensor = torch.zeros(max(k for k, v in probs_dict.items()) + 1)

        for key, value in probs_dict.items():
            probs_tensor[key] = value

        return int(torch.distributions.Categorical(probs_tensor).sample().item())

    def sampleNumMiddleNames(self):
        return int(torch.distributions.Categorical(torch.FloatTensor([1/3] * 3)).sample().item()) + 1

    def sample_aux(self, list: list, probs: torch.Tensor):
        sample = int(torch.distributions.Categorical(probs).sample().item())
        return list[sample]

    def generateMiddleInitial(self):
        initials = string.ascii_uppercase
        initials_sz = len(initials)
        probs = torch.FloatTensor([1/initials_sz] * initials_sz).to(DEVICE)
        sample = int(torch.distributions.Categorical(probs).sample().item())

        return initials[sample]
