# import sys
# sys.path.append("D:\OneDrive\Dokumenter\DTU\4. Sem\Fagprojekt\Deep_voice_conversion")

from zero_shot import Zero_shot as zero_shot
from download import download
from conversion import Instantiate_Models


if __name__ == "__main__":
    anders = "https://www.youtube.com/watch?v=_OW_vtWBRuw"
    s69 = "https://www.youtube.com/watch?v=PI1XZ0QADls"

    download("anders", anders, "musik_quiz/anders.wav")
    download("s69", s69, "musik_quiz/s69.wav")

    model, voc_model = Instantiate_Models(model_path = 'Models/autoVC_seed40_200k.pt')
    zero_shot("musik_quiz/s69.wav", "musik_quiz/anders.wav", model, voc_model, "musik_test")