"""Quick test runner for DroneVLA"""
import sys
sys.path.insert(0, '.')

import torch
from src.models.drone_vla import (
    VisualEncoder, StateEncoder, WorldModel, ActionDecoder, DroneVLA
)

def test_all():
    print('='*60)
    print('DroneVLA Model Tests')
    print('='*60)

    # Test 1: Visual Encoder
    print('\n[1/5] Testing Visual Encoder...')
    encoder = VisualEncoder(embed_dim=256, temporal=True)
    images = torch.randn(2, 4, 3, 64, 64)
    features = encoder(images)
    assert features.shape == (2, 256), f'Expected (2,256), got {features.shape}'
    print('  PASSED')

    # Test 2: State Encoder
    print('\n[2/5] Testing State Encoder...')
    state_enc = StateEncoder(state_dim=12, embed_dim=128)
    state = torch.randn(2, 12)
    state_feat = state_enc(state)
    assert state_feat.shape == (2, 128), f'Expected (2,128), got {state_feat.shape}'
    print('  PASSED')

    # Test 3: World Model
    print('\n[3/5] Testing World Model...')
    wm = WorldModel(state_dim=12, action_dim=4)
    state = torch.randn(2, 12)
    actions = torch.randn(2, 10, 4)
    future_states, future_rewards = wm.predict_future(state, actions)
    assert future_states.shape == (2, 10, 12)
    assert future_rewards.shape == (2, 10, 1)
    print('  PASSED')

    # Test 4: Action Decoder
    print('\n[4/5] Testing Action Decoder...')
    decoder = ActionDecoder(input_dim=256, action_dim=4, mode='deterministic')
    features = torch.randn(2, 256)
    action = decoder(features)
    assert action.shape == (2, 4)
    print('  PASSED (deterministic)')

    # Test 5: Full Model
    print('\n[5/5] Testing Full DroneVLA Model...')
    model = DroneVLA(
        visual_dim=256, language_dim=256, state_dim=12,
        state_embed_dim=128, action_dim=4, action_horizon=8,
        use_world_model=True, action_mode='deterministic'
    )
    images = torch.randn(2, 4, 3, 64, 64)
    instructions = ['hover at position 1,2,3', 'fly to the red building']
    state = torch.randn(2, 12)
    actions = torch.randn(2, 8, 4)
    outputs = model(images, instructions, state, actions)
    assert 'actions' in outputs
    assert 'visual_features' in outputs
    assert 'future_states' in outputs
    print('  PASSED')

    print('\n' + '='*60)
    print('ALL TESTS PASSED!')
    print('='*60)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel Parameters: {total_params:,}')

if __name__ == '__main__':
    test_all()
