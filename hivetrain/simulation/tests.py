import unittest
import torch
from unittest.mock import Mock, patch
from global_store import GlobalStore, YumaConsensus
from miner import Miner
from validator import Validator
from simulation import Simulation
from config import SimulationConfig

class TestGlobalStore(unittest.TestCase):
    def setUp(self):
        self.global_store = GlobalStore(num_validators=10, num_subnets=5)

    def test_update_miner_loss(self):
        miner_uid = 0
        self.global_store.update_miner_loss(miner_uid)
        self.assertIn(miner_uid, self.global_store.losses)
        self.assertEqual(len(self.global_store.losses[miner_uid]), 1)

    def test_get_miner_losses(self):
        miner_uid = 1
        self.global_store.update_miner_loss(miner_uid)
        losses = self.global_store.get_miner_losses(miner_uid)
        self.assertEqual(len(losses), 1)

    def test_update_global_mean(self):
        initial_mean = self.global_store.mean
        self.global_store.update_global_mean()
        self.assertLess(self.global_store.mean, initial_mean)

class TestYumaConsensus(unittest.TestCase):
    def setUp(self):
        self.yuma_consensus = YumaConsensus()

    def test_trust(self):
        W = torch.rand(10, 5)
        S = torch.rand(10)
        S = S / S.sum()  # Normalize
        T = self.yuma_consensus.trust(W, S)
        self.assertEqual(T.shape, torch.Size([5]))

    def test_rank(self):
        W = torch.rand(10, 5)
        S = torch.rand(10)
        S = S / S.sum()  # Normalize
        R = self.yuma_consensus.rank(W, S)
        self.assertEqual(R.shape, torch.Size([5]))
        self.assertAlmostEqual(R.sum().item(), 1.0, places=6)

    def test_consensus(self):
        T = torch.rand(5)
        C = self.yuma_consensus.consensus(T)
        self.assertEqual(C.shape, torch.Size([5]))
        self.assertTrue(all(0 <= x <= 1 for x in C))

    def test_emission(self):
        C = torch.rand(5)
        R = torch.rand(5)
        E = self.yuma_consensus.emission(C, R)
        self.assertEqual(E.shape, torch.Size([5]))
        self.assertAlmostEqual(E.sum().item(), 1.0, places=6)

class TestMiner(unittest.TestCase):
    def setUp(self):
        self.miner = Miner(uid=0)

    def test_update_loss(self):
        initial_loss = self.miner.current_loss
        self.miner.update_loss(5.0)
        self.assertEqual(self.miner.current_loss, 5.0)
        self.assertNotEqual(self.miner.current_loss, initial_loss)

    def test_get_loss_reduction(self):
        self.miner.update_loss(10.0)
        self.miner.update_loss(8.0)
        self.assertEqual(self.miner.get_loss_reduction(), 2.0)

class TestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = Validator(uid=0)

    def test_assign_miners(self):
        miner_uids = [1, 2, 3]
        self.validator.assign_miners(miner_uids)
        self.assertEqual(set(self.validator.assigned_miners), set(miner_uids))

    def test_evaluate_miners(self):
        self.validator.assign_miners([1, 2])
        global_store = {1: [10, 9], 2: [10, 11]}
        self.validator.evaluate_miners(global_store)
        self.assertEqual(len(self.validator.weights), 2)
        self.assertGreater(self.validator.weights[1], self.validator.weights[2])

class TestSimulation(unittest.TestCase):
    def setUp(self):
        config = SimulationConfig()
        self.simulation = Simulation(config)

    @patch('simulation.GlobalStore')
    @patch('simulation.Miner')
    @patch('simulation.Validator')
    def test_run_simulation(self, MockValidator, MockMiner, MockGlobalStore):
        self.simulation.run_simulation(num_rounds=5)
        self.assertEqual(MockGlobalStore().update_global_mean.call_count, 5)
        self.assertEqual(MockMiner().update_loss.call_count, 5 * self.simulation.num_miners)
        self.assertEqual(MockValidator().evaluate_miners.call_count, 5 * self.simulation.num_validators)

class TestSimulationConfig(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()

    def test_default_config(self):
        self.assertEqual(self.config.getint('SIMULATION', 'num_miners'), 100)
        self.assertEqual(self.config.getfloat('GLOBAL_STORE', 'initial_mean'), 10.0)

    def test_set_and_get(self):
        self.config.set('SIMULATION', 'num_miners', '200')
        self.assertEqual(self.config.getint('SIMULATION', 'num_miners'), 200)

    def test_save_and_load(self):
        self.config.set('SIMULATION', 'test_value', '42')
        self.config.save_config()
        new_config = SimulationConfig()
        self.assertEqual(new_config.get('SIMULATION', 'test_value'), '42')

if __name__ == '__main__':
    unittest.main()