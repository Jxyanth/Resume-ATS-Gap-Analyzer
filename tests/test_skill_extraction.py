import unittest

from ats_gap_analyzer import ATSGapAnalyzer


class StubPipeline:
    def __init__(self, entities):
        self.entities = entities

    def __call__(self, _text):
        return self.entities


class SkillExtractionTests(unittest.TestCase):
    def test_skill_spans_drop_prose_and_keep_separate_skill_names(self):
        analyzer = ATSGapAnalyzer(device="cpu")
        analyzer._ner_pipeline = StubPipeline(
            [
                {
                    "entity_group": "SKILLS",
                    "word": "Python, SQL and machine learning",
                    "score": 0.99,
                },
                {
                    "entity_group": "SKILLS",
                    "word": (
                        "AI solutions; and unstructured datasets for; concepts; learning; "
                        "manage; manager; models"
                    ),
                    "score": 0.99,
                },
                {
                    "entity_group": "Designation",
                    "word": "VP of Engineering key responsibilities",
                    "score": 0.99,
                },
                {
                    "entity_group": "Designation",
                    "word": "VP of Engineering",
                    "score": 0.99,
                },
                {
                    "entity_group": "Degree",
                    "word": "bachelor ’ s or",
                    "score": 0.99,
                },
                {
                    "entity_group": "Degree",
                    "word": "Bachelor of Science",
                    "score": 0.99,
                },
            ]
        )

        entities = analyzer.extract_entities("Requirements: Python and machine learning.")

        self.assertEqual(entities["skills"], ["machine learning", "python", "sql"])
        self.assertEqual(entities["designations"], ["vp of engineering"])
        self.assertEqual(entities["degrees"], ["bachelor of science"])


if __name__ == "__main__":
    unittest.main()
