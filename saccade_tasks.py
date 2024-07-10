import gymnasium as gym
from gymnasium import spaces
import numpy as np


class sDMS(gym.Env):
    '''
    Saccadic Delay Match to Sample task.

    Procedure:
        Sample Presentation: The subject fixates on a central point, and a sample stimulus briefly appears at a peripheral location.
        Delay Period: The sample stimulus is removed, and the subject continues to fixate on the central point during a delay.
        Intervening Stimuli: Multiple stimuli appear one by one at various peripheral locations, including the original sample stimulus.
        Choice Phase: The subject must make a saccade to the location of the original sample stimulus.

    Applications:
        Studies working memory and spatial recognition.
        Can be used to investigate the neural basis of memory retention and saccadic planning.
    '''

    def __init__(self, num_intervening=3, delay_period=10, sample_duration=5, intervening_duration=5):
        super(sDMS, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(
            low=0, high=360, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=360, shape=(1,), dtype=np.float32)

        # Task parameters
        self.num_intervening = num_intervening
        self.delay_period = delay_period
        self.sample_duration = sample_duration
        self.intervening_duration = intervening_duration

        # Internal state
        self.state = None
        self.sample_location = None
        self.intervening_locations = None
        self.current_step = 0
        self.current_intervening = 0
        self.phase = None

    def reset(self):
        self.current_step = 0
        self.current_intervening = 0
        self.sample_location = np.random.uniform(0, 360)

        # Generate random intervening stimuli and insert the sample stimulus randomly
        self.intervening_locations = np.random.uniform(
            0, 360, self.num_intervening - 1)
        insertion_index = np.random.randint(self.num_intervening)
        self.intervening_locations = np.insert(
            self.intervening_locations, insertion_index, self.sample_location)

        self.state = np.array([self.sample_location])
        self.phase = 'sample_presentation'

        return self.state

    def step(self, action):
        reward = 0
        done = False

        if self.phase == 'sample_presentation':
            self.current_step += 1
            if self.current_step >= self.sample_duration:
                self.phase = 'delay_period'
                self.current_step = 0
                self.state = np.array([0.0])  # Reset observation during delay
            else:
                self.state = np.array([self.sample_location])

        elif self.phase == 'delay_period':
            self.current_step += 1
            if self.current_step >= self.delay_period:
                self.phase = 'intervening_stimuli'
                self.current_step = 0
                self.state = np.array(
                    [self.intervening_locations[self.current_intervening]])
            else:
                self.state = np.array([0.0])  # Continue delay period

        elif self.phase == 'intervening_stimuli':
            self.current_step += 1
            if self.current_step >= self.intervening_duration:
                self.current_intervening += 1
                if self.current_intervening >= self.num_intervening:
                    self.phase = 'choice_phase'
                    # No stimulus shown, observation reset
                    self.state = np.array([0.0])
                else:
                    self.current_step = 0
                    self.state = np.array(
                        [self.intervening_locations[self.current_intervening]])

        elif self.phase == 'choice_phase':
            done = True
            action_angle = action[0]  # Extract the angle from action
            # Gaussian reward centered on sample_location
            reward = np.exp(-0.5 *
                            ((action_angle - self.sample_location) / 10)**2)

        info = {}
        return self.state, reward, done, info

    def render(self, mode='human'):
        if self.phase == 'sample_presentation':
            print(f"Sample at {self.sample_location} degrees")
        elif self.phase == 'delay_period':
            print("Delay period...")
        elif self.phase == 'intervening_stimuli':
            print(f"Intervening stimulus at {self.state[0]} degrees")
        elif self.phase == 'choice_phase':
            print("Make your choice.")


# Create the environment
env = sDMS()

# Example usage
obs = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

print(f"Total reward: {total_reward}")
