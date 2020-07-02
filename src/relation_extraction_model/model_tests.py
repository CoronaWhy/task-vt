import unittest
from re_model import RelationExtractModel
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent / "transformer_dir"
print(str(BASE_DIR))

class TestCoreMethods(unittest.TestCase):
    def setUp(self):
        self.model = RelationExtractModel(str(BASE_DIR), str(BASE_DIR / "weight.pickle"))

    def test_inference(self):
        self.assertTrue(self.model.predict("Here is some text", .5)[0])
        self.assertTrue(self.model.predict("A 1993 parallel-group study found evidence of suppressed short-term lower-leg growth, as measured by knemometry, with BUD (200 μg BID) or intramuscular methylprednisolone acetate (60 mg QD) when compared to terfenadine tablets (60 mg QD) in 44 children (aged 6-15 years) with AR prediction was", .3)[0])

    #def test_encode_text(self):
        #self.assertGreater(len(self.model.regular_encode(["This is some text to encode"], maxlen=100)), 2) 
        
if __name__ == '__main__':
    unittest.main()
