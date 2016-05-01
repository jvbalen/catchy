from __future__ import division

import numpy as np
import unittest

import pitch_features as pitch
import utils

class TestFeatures(unittest.TestCase):

    def setUp(self):
        data_dir = '/Users/Jan/Documents/Work/Cogitch/Data/HookedOnMusic/features/'
        pitch.melody_dir = data_dir + 'melody/'
        pitch.chroma_dir = data_dir + 'hpcp/'
        self.example_id = '147526770'

    def test_chroma_shape(self):
        t, chroma = pitch.get_chroma(self.example_id)
        self.assertEqual(chroma.shape[-1], 12)

    def test_feature_shapes(self):
        # non-interval-based features
        PH3 = pitch.get_pitchhist3(self.example_id, intervals=False)
        CH3 = pitch.get_chromahist3(self.example_id, intervals=False)
        PH2 = pitch.get_pitchhist2(self.example_id, intervals=False)
        CH2 = pitch.get_chromahist2(self.example_id, intervals=False)
        H = pitch.get_harmonisation(self.example_id, intervals=False)

        # interval-based features
        PH3_int = pitch.get_pitchhist3(self.example_id, intervals=True)
        CH3_int = pitch.get_chromahist3(self.example_id, intervals=True)
        PH2_int = pitch.get_pitchhist2(self.example_id, intervals=True)
        CH2_int = pitch.get_chromahist2(self.example_id, intervals=True) 
        H_int = pitch.get_harmonisation(self.example_id, intervals=True)

        self.assertTrue(np.all(PH3.shape == (12, 12, 12)))
        self.assertTrue(np.all(CH3.shape == (12, 12, 12)))
        self.assertTrue(np.all(PH2.shape == (12, 12)))
        self.assertTrue(np.all(CH2.shape == (12, 12)))
        self.assertTrue(np.all(H.shape == (12, 12)))

        self.assertTrue(np.all(PH3_int.shape == (12, 12)))
        self.assertTrue(np.all(CH3_int.shape == (12, 12)))
        self.assertTrue(np.all(PH2_int.shape == (12,)))
        self.assertTrue(np.all(CH2_int.shape == (12,)))
        self.assertTrue(np.all(H_int.shape == (12,)))


class TestTranspositionInvariance(unittest.TestCase):

    def setUp(self):
        data_dir = '/Users/Jan/Documents/Work/Cogitch/Data/HookedOnMusic/features/'
        pitch.chroma_dir = data_dir + 'hpcp/'
        self.example_id = '147526770'

        t, chroma = pitch.get_chroma(self.example_id)
        print chroma.shape, '\n'
        chroma_trans = np.roll(chroma, -3, axis=1)
        print chroma_trans.shape, '\n'

        self.temp_dir = '/Users/Jan/Documents/Work/Cogitch/Data/HookedOnMusic/features/temp/'
        utils.write_feature([t, chroma_trans], [self.temp_dir, self.example_id])

    def test_invariance(self):
        CH3 = pitch.get_chromahist3(self.example_id, intervals=False)
        CH3_int = pitch.get_chromahist3(self.example_id, intervals=True)
        
        pitch.chroma_dir = self.temp_dir
        CH3_trans = pitch.get_chromahist3(self.example_id, intervals=False)
        CH3_int_trans = pitch.get_chromahist3(self.example_id, intervals=True)

        self.assertTrue(not np.allclose(CH3, CH3_trans))
        self.assertTrue(np.allclose(CH3_int, CH3_int_trans))


if __name__ == '__main__':
    unittest.main()


