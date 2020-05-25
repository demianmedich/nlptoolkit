# coding=utf-8
import itertools
import os
import unittest

from torch.utils.data.dataloader import DataLoader

from nlptoolkit.dataset.text import IterableTextLineDataset
from nlptoolkit.dataset.text import StringToTensor
from nlptoolkit.tokenizer import BertWordPieceTokenizer


class DatasetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_files = [
            "__temp_sample_001.txt",
            "__temp_sample_002.txt",
            "__temp_sample_003.txt",
        ]
        self.contents = [
            """The Witcher 3: Wild Hunt is a 2015 action role-playing game developed and published by Polish developer CD Projekt and is based on The Witcher series of fantasy novels by Andrzej Sapkowski.
It is the sequel to the 2011 game The Witcher 2: Assassins of Kings, played in an open world with a third-person perspective.
Players control protagonist Geralt of Rivia, a monster slayer (known as a Witcher) who is looking for his missing adopted daughter on the run from the Wild Hunt, an otherworldly force determined to capture her and use her powers.
Players battle the game's many dangers with weapons and magic, interact with non-player characters, and complete main-story and side quests to acquire experience points and gold, which are used to increase Geralt's abilities and purchase equipment.
Its central story has several endings, determined by the player's choices at certain points in the game.
Development began in 2011 and lasted for three and a half years.
Voice recording took more than two and a half years.
The writing was infused with realistic aspects such as moral ambiguity in an attempt to avoid simplification, impart authenticity, and reflect Sapkowski's novels.
Central and Northern European cultures formed the basis of the game's world.
REDengine 3 enabled the developer to create a complex story without compromising the game's open world.
The music was composed by Marcin Przybyłowicz and performed by the Brandenburg State Orchestra.
The Witcher 3: Wild Hunt was released for Microsoft Windows, PlayStation 4, and Xbox One in May 2015, with a Nintendo Switch version released in October 2019.
The game received critical acclaim, with praise for its gameplay, narrative, world design, combat, and visuals, although it received minor criticism due to technical issues.
It received numerous Game of the Year awards and has been cited as one of the best video games ever made.
Two expansions were also released to critical acclaim: Hearts of Stone and Blood and Wine.
A Game of the Year edition was released in August 2016, with the base game, expansions, and all downloadable content.
It was a commercial success, with the game and its expansions shipping over 28 million copies.""",
            """Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).
It was first identified in December 2019 in Wuhan, China, and has since spread globally, resulting in an ongoing pandemic.
As of 23 May 2020, more than 5.22 million cases have been reported across 188 countries and territories, resulting in more than 338,000 deaths.
More than 2.06 million people have recovered.
Common symptoms include fever, cough, fatigue, shortness of breath, and loss of smell and taste.
While the majority of cases result in mild symptoms, some progress to acute respiratory distress syndrome (ARDS) likely precipitated by cytokine storm, multi-organ failure, septic shock, and blood clots.
The time from exposure to onset of symptoms is typically around five days but may range from two to fourteen days.
The virus is primarily spread between people during close contact, most often via small droplets produced by coughing, sneezing, and talking.
The droplets usually fall to the ground or onto surfaces rather than travelling through air over long distances.
Less commonly, people may become infected by touching a contaminated surface and then touching their face.
It is most contagious during the first three days after the onset of symptoms, although spread is possible before symptoms appear, and from people who do not show symptoms.
The standard method of diagnosis is by real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab.
Chest CT imaging may also be helpful for diagnosis in individuals where there is a high suspicion of infection based on symptoms and risk factors; however, guidelines do not recommend using CT imaging for routine screening.
Recommended measures to prevent infection include frequent hand washing, maintaining physical distance from others (especially from those with symptoms), quarantine (especially for those with symptoms), covering coughs, and keeping unwashed hands away from the face.
In addition, the use of a face covering is recommended for those who suspect they have the virus and their caregivers.
Recommendations for face covering use by the general public vary, with some authorities recommending for them, some recommending against them (to conserve masks for healthcare workers), and others requiring their use.
There is limited evidence for or against the use of masks (medical or other) in healthy individuals in the wider community.
According to the World Health Organization, there are no available vaccines nor specific antiviral treatments for COVID-19.
On 1 May 2020, the United States gave Emergency Use Authorization to the antiviral remdesivir for people hospitalized with severe COVID‑19.
Management involves the treatment of symptoms, supportive care, isolation, and experimental measures.
The World Health Organization (WHO) declared the COVID‑19 outbreak a Public Health Emergency of International Concern (PHEIC) on 30 January 2020 and a pandemic on 11 March 2020.
Local transmission of the disease has occurred in most countries across all six WHO regions.""",
            """PyTorch Datasets are objects that have a single job: to return a single datapoint on request.
The exact form of the datapoint varies between tasks: it could be a single image, a slice of a time series, a tabular record or something else entirely. 
These are then passed on to a Dataloader which handles batching of datapoints and parallelism.
Before PyTorch 1.2 the only available dataset class was the original “map-style” dataset. 
This simply requires the user to inherit from the torch.utils.data.Dataset class and implement the __len__ and __getitem__ methods, where __getitem__ receives an index which is mapped to some item in your dataset.
Let’s see a very simple example.
This is instantiated and passed to the DataLoader, which is iterated over, returning batches of data to feed into our model.
This remains a flexible abstraction, however, the assumption that you can trivially map each data point in your dataset means that it is less suited to situations where the input data is arriving as part of a stream, for example, an audio or video feed.
Alternatively, each datapoint might be a subset of a file which is too large to be held in memory and so requires incremental loading during training.
These situations can be addressed with more complex logic in our dataset or additional pre-processing of our inputs, but there is now a more natural solution, enter the IterableDataset!""",
        ]
        for i, sample_file in enumerate(self.sample_files):
            with open(sample_file, mode="wt", encoding="utf-8") as f:
                for line in self.contents[i].split("\n"):
                    f.write(f"{line}\n")

    def tearDown(self) -> None:
        for sample_file in self.sample_files:
            os.remove(sample_file)

    def test_iterable_text_line_dataset(self):
        corpus_files = self.sample_files
        vocab_file = "../../vocab/bert_wordpiece_en_cased_vocab.txt"
        tokenizer = BertWordPieceTokenizer(vocab_file, do_lower_case=False)
        max_seq_len = 512
        # tokenizer = None
        ds = IterableTextLineDataset(corpus_files,
                                     num_replicas=2,
                                     rank=0,
                                     infinite=True,
                                     shuffle_files=False)
        collate_fn = StringToTensor(tokenizer, max_seq_len)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

        for i, batch in itertools.islice(enumerate(loader), 100):
            print(f"{i} entry\n  {batch}")


if __name__ == '__main__':
    unittest.main()
