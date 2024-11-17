import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Spiro:
    def __init__(self, x, y, vx, vy, size):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.size = size
    
    def move(self, time_step):
        """Update the position based on velocity."""
        self.position += self.velocity * time_step
    
    def reflect(self, width, height):
        """Reflect the velocity vector when the particle hits the boundary."""
        # Reflect on the x boundary
        if self.position[0] <= 0 or self.position[0] >= width:
            self.velocity[0] = -self.velocity[0]
        
        # Reflect on the y boundary
        if self.position[1] <= 0 or self.position[1] >= height:
            self.velocity[1] = -self.velocity[1]
    def apply_periodic_boundary(self, width, height):
        """
        Apply periodic boundary conditions to the particle.
        If the particle moves out of the box, it re-enters from the opposite side.
        """
        self.position[0] %= width  # Wrap around in the x-direction
        self.position[1] %= height  # Wrap around in the y-direction
    
    def get_position(self):
        """Returns the current position of the particle."""
        return self.position

    def update_velocity(self, neighbors, noise_std=0.25, max_speed=0.5):
        """Update velocity based on the average velocity of neighboring spiros."""
        if neighbors:  # Ensure there are neighbors to average with
            # Collect velocities of self and neighbors
            velocities = [self.velocity] + [neighbor.velocity for neighbor in neighbors]
            
            # Calculate the average velocity vector
            average_velocity = np.mean(velocities, axis=0)
            
            # Normalize to get the average direction
            avg_direction = average_velocity / np.linalg.norm(average_velocity)
            
            # Keep the original speed (magnitude of the current velocity)
            speed = np.linalg.norm(self.velocity)
            
            # Update velocity to the new direction with the original speed
            self.velocity = avg_direction * speed
            # average_velocity = np.mean([neighbor.velocity for neighbor in neighbors], axis=0)
            # num_neighbors = len(neighbors)
            # self.velocity = average_velocity*(num_neighbors/(num_neighbors+1)) + (self.velocity/(num_neighbors+1))  # Blend current and average velocity
            noise = np.random.normal(loc=0.0, scale=noise_std, size=self.velocity.shape)
            self.velocity += noise

            current_speed = np.linalg.norm(self.velocity)
            if current_speed > max_speed:
                self.velocity = (self.velocity / current_speed) * max_speed
        
def testing():
    print("testing from simulation.py")

class Simulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.positions = None

    def initialize_spiros(self, num_particles):
        self.particles.clear()
        for _ in range(num_particles):
            x = np.random.uniform(1, self.width-1)
            y = np.random.uniform(1, self.height-1)
            vx = np.random.uniform(-.7, .7)
            vy = np.random.uniform(-.7, .7)
            size = .1 # np.random.uniform(0.05, 0.1)
            self.particles.append(Spiro(x, y, vx, vy, size))

    def find_neighbors(self, spiro, radius):
        """Find neighboring particles within a certain radius."""
        neighbors = []
        for particle in self.particles:
            if particle is not spiro:  # Exclude itself
                distance = np.linalg.norm(spiro.position - particle.position)
                if distance < radius:
                    neighbors.append(particle)
        return neighbors

    # def run_simulation(self, num_steps, time_step):
    #     num_particles = len(self.particles)
        
    #     # Simulation loop to update particle positions and handle reflections
    #     self.positions = np.zeros((num_particles, num_steps, 2))  # To store positions for visualization
    #     for step in range(num_steps):
    #         for i, particle in enumerate(self.particles):
    #             particle.move(time_step)
    #             particle.reflect(self.width, self.height)
    #             self.positions[i, step] = particle.get_position()

    def run_simulation(self, num_steps, time_step, interaction_radius):
        num_particles = len(self.particles)
        
        # Simulation loop to update particle positions and handle reflections
        self.positions = np.zeros((num_particles, num_steps, 2))  # To store positions for visualization
        for step in range(num_steps):
            # Update velocity based on neighboring particles
            for particle in self.particles:
                neighbors = self.find_neighbors(particle, interaction_radius)
                particle.update_velocity(neighbors)
            
            # Move particles and handle boundary reflections
            for i, particle in enumerate(self.particles):
                particle.move(time_step)
                particle.apply_periodic_boundary(self.width, self.height)
                self.positions[i, step] = particle.get_position()

    def plot_trajectory(self):
        # Plotting the trajectory of particles
        plt.figure(figsize=(10, 8))
        for i in range(len(self.particles)):
            plt.plot(self.positions[i, :, 0], self.positions[i, :, 1], lw=0.8)
    
        # Customize plot
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title('2D Trajectories of Particles (Spiro)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def animate(self, num_steps, time_step, interaction_radius):
        """Create an animation of the particle trajectories."""
        self.run_simulation(num_steps, time_step, interaction_radius)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Particle Motion with Interaction')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Scatter plot for particles
        scat = ax.scatter([], [], s=50, color='blue')

        def update(frame):
            positions = self.positions[:, frame, :]
            scat.set_offsets(positions)
            return scat,

        anim = FuncAnimation(fig, update, frames=num_steps, interval=30, blit=True)
        plt.close(fig)
        
        # Save animation as an mp4 or gif file
        anim.save("particle_simulation.gif", writer="pillow", dpi=80)
        print("done outputing animation")