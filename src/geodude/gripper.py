"""Gripper control and grasp detection."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager, detect_grasped_object

# Gripper joint trajectory recorded from physics simulation.
# Shape: (101, 8) - 101 waypoints from open (t=0) to closed (t=1), 8 joints.
# Joint order: left_coupler, left_driver, left_follower, left_spring_link,
#              right_coupler, right_driver, right_follower, right_spring_link
_GRIPPER_JOINT_NAMES = [
    "left_coupler", "left_driver", "left_follower", "left_spring_link",
    "right_coupler", "right_driver", "right_follower", "right_spring_link",
]
_GRIPPER_TRAJECTORY = np.array([
    [-0.000622, -0.025613, 0.032463, -0.027563, -0.000610, -0.025589, 0.032481, -0.027446],
    [-0.000622, -0.028316, 0.035389, -0.030293, -0.000609, -0.028292, 0.035394, -0.030171],
    [-0.000621, -0.032757, 0.040187, -0.034784, -0.000609, -0.032733, 0.040171, -0.034654],
    [-0.000621, -0.038365, 0.046238, -0.040459, -0.000608, -0.038340, 0.046197, -0.040319],
    [-0.000620, -0.044755, 0.053128, -0.046928, -0.000608, -0.044731, 0.053058, -0.046777],
    [-0.000619, -0.051671, 0.060579, -0.053929, -0.000607, -0.051646, 0.060479, -0.053766],
    [-0.000619, -0.058939, 0.068403, -0.061287, -0.000606, -0.058914, 0.068272, -0.061112],
    [-0.000618, -0.066444, 0.076475, -0.068885, -0.000606, -0.066419, 0.076312, -0.068697],
    [-0.000618, -0.074108, 0.084712, -0.076643, -0.000605, -0.074083, 0.084518, -0.076442],
    [-0.000617, -0.081879, 0.093057, -0.084508, -0.000604, -0.081854, 0.092831, -0.084294],
    [-0.000616, -0.089721, 0.101472, -0.092445, -0.000603, -0.089696, 0.101216, -0.092217],
    [-0.000616, -0.097612, 0.109932, -0.100429, -0.000603, -0.097586, 0.109646, -0.100189],
    [-0.000615, -0.105534, 0.118421, -0.108446, -0.000602, -0.105508, 0.118105, -0.108192],
    [-0.000614, -0.113479, 0.126926, -0.116483, -0.000601, -0.113453, 0.126582, -0.116216],
    [-0.000614, -0.121438, 0.135441, -0.124535, -0.000601, -0.121411, 0.135069, -0.124255],
    [-0.000613, -0.129407, 0.143961, -0.132595, -0.000600, -0.129380, 0.143561, -0.132302],
    [-0.000612, -0.137382, 0.152481, -0.140662, -0.000599, -0.137355, 0.152055, -0.140355],
    [-0.000612, -0.145362, 0.161001, -0.148732, -0.000598, -0.145335, 0.160548, -0.148412],
    [-0.000611, -0.153345, 0.169517, -0.156804, -0.000598, -0.153318, 0.169039, -0.156472],
    [-0.000611, -0.161330, 0.178031, -0.164877, -0.000597, -0.161302, 0.177528, -0.164532],
    [-0.000610, -0.169316, 0.186540, -0.172951, -0.000596, -0.169289, 0.186013, -0.172593],
    [-0.000609, -0.177303, 0.195044, -0.181025, -0.000596, -0.177276, 0.194495, -0.180655],
    [-0.000609, -0.185291, 0.203544, -0.189100, -0.000595, -0.185263, 0.202972, -0.188716],
    [-0.000608, -0.193279, 0.212040, -0.197173, -0.000594, -0.193251, 0.211445, -0.196777],
    [-0.000607, -0.201268, 0.220530, -0.205247, -0.000593, -0.201240, 0.219914, -0.204838],
    [-0.000607, -0.209257, 0.229015, -0.213320, -0.000593, -0.209229, 0.228379, -0.212898],
    [-0.000606, -0.217246, 0.237496, -0.221392, -0.000592, -0.217218, 0.236839, -0.220958],
    [-0.000606, -0.225235, 0.245972, -0.229464, -0.000591, -0.225207, 0.245296, -0.229017],
    [-0.000605, -0.233224, 0.254443, -0.237535, -0.000591, -0.233196, 0.253748, -0.237076],
    [-0.000604, -0.241214, 0.262909, -0.245605, -0.000590, -0.241185, 0.262196, -0.245134],
    [-0.000604, -0.249203, 0.271371, -0.253675, -0.000589, -0.249174, 0.270640, -0.253192],
    [-0.000603, -0.257193, 0.279828, -0.261745, -0.000589, -0.257164, 0.279080, -0.261249],
    [-0.000602, -0.265182, 0.288280, -0.269814, -0.000588, -0.265153, 0.287517, -0.269305],
    [-0.000602, -0.273172, 0.296728, -0.277882, -0.000587, -0.273143, 0.295949, -0.277361],
    [-0.000601, -0.281162, 0.305172, -0.285950, -0.000586, -0.281132, 0.304377, -0.285416],
    [-0.000601, -0.289151, 0.313612, -0.294017, -0.000586, -0.289122, 0.312802, -0.293471],
    [-0.000600, -0.297141, 0.322047, -0.302084, -0.000585, -0.297112, 0.321223, -0.301526],
    [-0.000599, -0.305131, 0.330478, -0.310150, -0.000584, -0.305101, 0.329641, -0.309579],
    [-0.000599, -0.313121, 0.338905, -0.318215, -0.000584, -0.313091, 0.338055, -0.317633],
    [-0.000598, -0.321111, 0.347327, -0.326280, -0.000583, -0.321081, 0.346466, -0.325685],
    [-0.000597, -0.329101, 0.355746, -0.334345, -0.000582, -0.329070, 0.354873, -0.333738],
    [-0.000597, -0.337091, 0.364161, -0.342409, -0.000582, -0.337060, 0.363276, -0.341790],
    [-0.000596, -0.345081, 0.372572, -0.350473, -0.000581, -0.345050, 0.371677, -0.349841],
    [-0.000596, -0.353071, 0.380979, -0.358536, -0.000580, -0.353040, 0.380074, -0.357892],
    [-0.000595, -0.361061, 0.389382, -0.366599, -0.000580, -0.361030, 0.388468, -0.365942],
    [-0.000594, -0.369051, 0.397782, -0.374661, -0.000579, -0.369020, 0.396859, -0.373992],
    [-0.000594, -0.377041, 0.406178, -0.382723, -0.000578, -0.377010, 0.405246, -0.382042],
    [-0.000593, -0.385031, 0.414570, -0.390784, -0.000578, -0.385000, 0.413631, -0.390091],
    [-0.000592, -0.393021, 0.422959, -0.398845, -0.000577, -0.392990, 0.422012, -0.398140],
    [-0.000592, -0.401011, 0.431344, -0.406906, -0.000576, -0.400980, 0.430391, -0.406188],
    [-0.000591, -0.409002, 0.439725, -0.414966, -0.000576, -0.408970, 0.438766, -0.414236],
    [-0.000590, -0.416992, 0.448103, -0.423025, -0.000575, -0.416961, 0.447139, -0.422283],
    [-0.000590, -0.424982, 0.456478, -0.431085, -0.000574, -0.424951, 0.455509, -0.430330],
    [-0.000589, -0.432973, 0.464850, -0.439144, -0.000574, -0.432941, 0.463876, -0.438377],
    [-0.000589, -0.440963, 0.473218, -0.447202, -0.000573, -0.440931, 0.472240, -0.446423],
    [-0.000588, -0.448953, 0.481582, -0.455261, -0.000572, -0.448922, 0.480602, -0.454469],
    [-0.000587, -0.456944, 0.489944, -0.463318, -0.000571, -0.456912, 0.488960, -0.462515],
    [-0.000587, -0.464934, 0.498302, -0.471376, -0.000571, -0.464902, 0.497316, -0.470560],
    [-0.000586, -0.472925, 0.506657, -0.479433, -0.000570, -0.472893, 0.505670, -0.478605],
    [-0.000585, -0.480915, 0.515009, -0.487490, -0.000569, -0.480883, 0.514021, -0.486649],
    [-0.000585, -0.488906, 0.523358, -0.495547, -0.000569, -0.488873, 0.522369, -0.494693],
    [-0.000584, -0.496896, 0.531704, -0.503603, -0.000568, -0.496864, 0.530715, -0.502737],
    [-0.000583, -0.504887, 0.540047, -0.511659, -0.000567, -0.504854, 0.539059, -0.510781],
    [-0.000583, -0.512877, 0.548387, -0.519714, -0.000567, -0.512845, 0.547400, -0.518824],
    [-0.000582, -0.520868, 0.556723, -0.527770, -0.000566, -0.520835, 0.555738, -0.526867],
    [-0.000581, -0.528859, 0.565057, -0.535825, -0.000565, -0.528826, 0.564074, -0.534909],
    [-0.000581, -0.536849, 0.573388, -0.543880, -0.000565, -0.536817, 0.572408, -0.542951],
    [-0.000580, -0.544840, 0.581716, -0.551934, -0.000564, -0.544807, 0.580740, -0.550993],
    [-0.000579, -0.552831, 0.590041, -0.559989, -0.000563, -0.552798, 0.589069, -0.559035],
    [-0.000579, -0.560821, 0.598364, -0.568043, -0.000562, -0.560788, 0.597396, -0.567076],
    [-0.000578, -0.568812, 0.606683, -0.576096, -0.000562, -0.568779, 0.605720, -0.575117],
    [-0.000577, -0.576803, 0.615000, -0.584150, -0.000561, -0.576770, 0.614043, -0.583158],
    [-0.000576, -0.584794, 0.623314, -0.592203, -0.000560, -0.584760, 0.622363, -0.591198],
    [-0.000576, -0.592784, 0.631625, -0.600256, -0.000560, -0.592751, 0.630681, -0.599238],
    [-0.000575, -0.600775, 0.639934, -0.608309, -0.000559, -0.600742, 0.638998, -0.607278],
    [-0.000574, -0.608766, 0.648239, -0.616362, -0.000558, -0.608733, 0.647311, -0.615318],
    [-0.000574, -0.616757, 0.656542, -0.624414, -0.000558, -0.616724, 0.655623, -0.623357],
    [-0.000573, -0.624748, 0.664843, -0.632467, -0.000557, -0.624714, 0.663933, -0.631396],
    [-0.000572, -0.632739, 0.673141, -0.640519, -0.000556, -0.632705, 0.672241, -0.639435],
    [-0.000571, -0.640730, 0.681436, -0.648571, -0.000555, -0.640696, 0.680547, -0.647474],
    [-0.000571, -0.648721, 0.689728, -0.656622, -0.000555, -0.648687, 0.688851, -0.655512],
    [-0.000570, -0.656712, 0.698018, -0.664674, -0.000554, -0.656678, 0.697152, -0.663550],
    [-0.000569, -0.664703, 0.706306, -0.672726, -0.000553, -0.664669, 0.705452, -0.671588],
    [-0.000568, -0.672694, 0.714590, -0.680777, -0.000552, -0.672660, 0.713750, -0.679626],
    [-0.000568, -0.680685, 0.722873, -0.688828, -0.000552, -0.680651, 0.722046, -0.687663],
    [-0.000567, -0.688676, 0.731152, -0.696879, -0.000551, -0.688642, 0.730340, -0.695701],
    [-0.000566, -0.696667, 0.739429, -0.704930, -0.000550, -0.696633, 0.738633, -0.703738],
    [-0.000565, -0.704658, 0.747704, -0.712981, -0.000549, -0.704624, 0.746923, -0.711775],
    [-0.000565, -0.712649, 0.755976, -0.721031, -0.000549, -0.712615, 0.755212, -0.719811],
    [-0.000564, -0.720640, 0.764246, -0.729082, -0.000548, -0.720606, 0.763498, -0.727848],
    [-0.000563, -0.728631, 0.772513, -0.737133, -0.000547, -0.728597, 0.771783, -0.735884],
    [-0.000562, -0.736622, 0.780777, -0.745183, -0.000546, -0.736588, 0.780066, -0.743920],
    [-0.000561, -0.744613, 0.789039, -0.753233, -0.000546, -0.744579, 0.788348, -0.751956],
    [-0.000561, -0.752604, 0.797299, -0.761284, -0.000545, -0.752570, 0.796627, -0.759992],
    [-0.000560, -0.760596, 0.805556, -0.769334, -0.000544, -0.760561, 0.804905, -0.768028],
    [-0.000559, -0.768587, 0.813810, -0.777384, -0.000543, -0.768553, 0.813181, -0.776063],
    [-0.000558, -0.776578, 0.822062, -0.785434, -0.000542, -0.776544, 0.821455, -0.784099],
    [-0.000557, -0.784569, 0.830312, -0.793484, -0.000542, -0.784535, 0.829728, -0.792134],
    [-0.000515, -0.792415, 0.838404, -0.801223, -0.000540, -0.792402, 0.837864, -0.800002],
    [-0.000416, -0.797107, 0.843477, -0.805558, -0.000418, -0.797098, 0.842906, -0.804302],
    [-0.000420, -0.800270, 0.846809, -0.808623, -0.000422, -0.800264, 0.846252, -0.807362],
])


class Gripper:
    """Controls a gripper and detects grasp state.

    Handles:
    - Opening and closing the gripper via MuJoCo actuator
    - Detecting when an object is grasped (via contacts or geometry)
    - Updating grasp state in GraspManager
    - Kinematic attachment for non-physics execution

    For physics-based execution, use open()/close() which simulate gripper motion.
    For kinematic execution, use kinematic_open()/kinematic_close() which set
    gripper position directly and use geometric grasp detection.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        actuator_name: str,
        gripper_body_names: list[str],
        grasp_manager: GraspManager,
        ctrl_open: float = 0.0,
        ctrl_closed: float = 255.0,
        gripper_site_name: str | None = None,
    ):
        """Initialize gripper controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            arm_name: Name of arm this gripper belongs to ("left" or "right")
            actuator_name: Name of gripper actuator in MuJoCo
            gripper_body_names: Names of gripper bodies for contact detection
            grasp_manager: GraspManager for tracking grasp state
            ctrl_open: Control value for open gripper
            ctrl_closed: Control value for closed gripper
            gripper_site_name: Name of gripper site for pose queries and attachment
        """
        self.model = model
        self.data = data
        self.arm_name = arm_name
        self.gripper_body_names = gripper_body_names
        self.grasp_manager = grasp_manager
        self.ctrl_open = ctrl_open
        self.ctrl_closed = ctrl_closed
        self.gripper_site_name = gripper_site_name

        # Get actuator index
        if actuator_name:
            self.actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if self.actuator_id == -1:
                raise ValueError(f"Actuator '{actuator_name}' not found in model")
        else:
            self.actuator_id = None

        self._candidate_objects: list[str] | None = None
        # Grasp detection threshold for kinematic mode (distance in meters)
        self.kinematic_grasp_threshold = 0.05

        # Find ALL gripper joint indices for kinematic control
        # Robotiq 2F-140 has: driver, coupler, follower, spring_link joints on each side
        # Map from joint name to qpos index, ordered to match _GRIPPER_JOINT_NAMES
        self._gripper_joint_qpos_indices: list[int] = []
        if actuator_name:
            prefix = "/".join(actuator_name.split("/")[:-1])
            for joint_name in _GRIPPER_JOINT_NAMES:
                full_name = f"{prefix}/{joint_name}_joint"
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                if joint_id != -1:
                    self._gripper_joint_qpos_indices.append(model.jnt_qposadr[joint_id])

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        """Set the list of objects that could be grasped.

        This helps grasp detection by limiting which bodies are considered.

        Args:
            objects: List of object body names, or None to consider all
        """
        self._candidate_objects = objects

    def set_kinematic_position(self, t: float) -> None:
        """Set gripper to interpolated position kinematically (no physics).

        Uses pre-recorded trajectory from physics simulation to ensure
        correct 4-bar linkage geometry at any position.

        Args:
            t: Position from 0.0 (open) to 1.0 (closed)
        """
        if not self._gripper_joint_qpos_indices:
            return

        t = np.clip(t, 0.0, 1.0)

        # Interpolate in the trajectory
        n = len(_GRIPPER_TRAJECTORY) - 1
        idx = t * n
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n)
        alpha = idx - idx_low

        # Linear interpolation between waypoints
        joint_positions = (1 - alpha) * _GRIPPER_TRAJECTORY[idx_low] + alpha * _GRIPPER_TRAJECTORY[idx_high]

        # Set all joint positions
        for i, qpos_idx in enumerate(self._gripper_joint_qpos_indices):
            self.data.qpos[qpos_idx] = joint_positions[i]

        mujoco.mj_forward(self.model, self.data)

    def set_kinematic_open(self) -> None:
        """Set gripper to open position kinematically (no physics)."""
        self.set_kinematic_position(0.0)

    def set_kinematic_closed(self) -> None:
        """Set gripper to closed position kinematically (no physics)."""
        self.set_kinematic_position(1.0)

    def open(self, steps: int = 100) -> None:
        """Open the gripper.

        Args:
            steps: Number of simulation steps to run after commanding open
        """
        if self.actuator_id is None:
            return

        # Release any currently grasped objects first
        for obj in self.grasp_manager.get_grasped_by(self.arm_name):
            self.grasp_manager.mark_released(obj)

        # Command gripper open
        self.data.ctrl[self.actuator_id] = self.ctrl_open

        # Step simulation to let gripper open
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def close(self, steps: int = 100) -> str | None:
        """Close the gripper and detect grasp.

        Args:
            steps: Number of simulation steps to run after commanding close

        Returns:
            Name of grasped object, or None if nothing grasped
        """
        if self.actuator_id is None:
            return None

        # Command gripper closed
        self.data.ctrl[self.actuator_id] = self.ctrl_closed

        # Step simulation to let gripper close
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        # Detect what we grasped
        grasped = detect_grasped_object(
            self.model,
            self.data,
            self.gripper_body_names,
            self._candidate_objects,
        )

        # Update grasp state
        if grasped:
            self.grasp_manager.mark_grasped(grasped, self.arm_name)

        return grasped

    def get_position(self) -> float:
        """Get current gripper TARGET position (0=open, 1=closed).

        Note: This returns the control target, not the actual position.
        Use get_actual_position() for the actual joint position.
        """
        if self.actuator_id is None:
            return 0.0

        ctrl = self.data.ctrl[self.actuator_id]
        return (ctrl - self.ctrl_open) / (self.ctrl_closed - self.ctrl_open)

    def get_actual_position(self) -> float:
        """Get actual gripper position from joint qpos (0=open, 1=closed).

        Reads the actual joint positions and estimates the gripper opening
        by comparing to the known open/closed trajectory waypoints.
        """
        if not self._gripper_joint_qpos_indices:
            return 0.0

        # Read actual joint positions
        actual = np.array([self.data.qpos[idx] for idx in self._gripper_joint_qpos_indices])

        # Compare to open and closed positions from trajectory
        open_pos = _GRIPPER_TRAJECTORY[0]
        closed_pos = _GRIPPER_TRAJECTORY[-1]

        # Use the driver joint (index 1, "right_driver") as reference - it has the most range
        # Driver goes from ~-0.026 (open) to ~-0.241 (closed)
        driver_idx = 1
        driver_actual = actual[driver_idx]
        driver_open = open_pos[driver_idx]
        driver_closed = closed_pos[driver_idx]

        # Compute normalized position (0=open, 1=closed)
        if abs(driver_closed - driver_open) < 1e-6:
            return 0.0
        t = (driver_actual - driver_open) / (driver_closed - driver_open)
        return float(np.clip(t, 0.0, 1.0))

    def set_position(self, position: float) -> None:
        """Set gripper position (0=open, 1=closed)."""
        if self.actuator_id is None:
            return

        ctrl = self.ctrl_open + position * (self.ctrl_closed - self.ctrl_open)
        self.data.ctrl[self.actuator_id] = ctrl

    @property
    def is_holding(self) -> bool:
        """Check if gripper is currently holding an object."""
        return len(self.grasp_manager.get_grasped_by(self.arm_name)) > 0

    @property
    def held_object(self) -> str | None:
        """Get the name of the currently held object, or None."""
        held = self.grasp_manager.get_grasped_by(self.arm_name)
        return held[0] if held else None

    def _check_gripper_contact(self) -> str | None:
        """Check if gripper is in contact with any non-gripper body.

        Stops on contact with anything - objects, robot base, etc.

        Returns:
            Name of contacted body, or None
        """
        # Get gripper body IDs (bodies we should ignore contact with)
        gripper_body_ids = set()
        for body_name in self.gripper_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                gripper_body_ids.add(body_id)

        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]

            # Check if one is gripper, one is something else
            if geom1_body in gripper_body_ids and geom2_body not in gripper_body_ids:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)
                return body_name
            if geom2_body in gripper_body_ids and geom1_body not in gripper_body_ids:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
                return body_name

        return None

    def kinematic_close(self, steps: int = 50) -> str | None:
        """Close gripper kinematically until contact.

        Incrementally closes the gripper, stopping when contact is detected
        with anything (object, robot base, etc.). Uses pre-recorded trajectory
        for correct 4-bar linkage geometry.

        Only grasps and attaches the contacted body if it's a valid graspable
        object (in candidate_objects list if specified, or a free body).

        Args:
            steps: Number of incremental steps to check for contact

        Returns:
            Name of grasped object, or None if nothing grasped
        """
        if not self._gripper_joint_qpos_indices:
            return None

        contacted_body = None

        # Incrementally close until contact with anything
        for i in range(steps + 1):
            t = i / steps
            self.set_kinematic_position(t)

            # Check for contact with any non-gripper body
            contacted_body = self._check_gripper_contact()
            if contacted_body:
                break

        # Determine if contacted body is graspable
        grasped = None
        if contacted_body:
            if self._candidate_objects is None or contacted_body in self._candidate_objects:
                # It's a valid graspable object
                grasped = contacted_body

        # If no contact detected, also check geometric proximity as fallback
        if not grasped and not contacted_body:
            grasped = self._detect_grasp_geometric()

        if grasped:
            # Update grasp state (collision groups)
            self.grasp_manager.mark_grasped(grasped, self.arm_name)

            # Attach object for kinematic movement
            attach_body = self.gripper_body_names[0] if self.gripper_body_names else None
            if attach_body:
                self.grasp_manager.attach_object(grasped, attach_body)

        return grasped

    def kinematic_open(self) -> None:
        """Open gripper kinematically and release any attached object.

        Sets gripper joint positions directly without physics simulation.
        """
        # Release any currently grasped objects
        for obj in self.grasp_manager.get_grasped_by(self.arm_name):
            self.grasp_manager.detach_object(obj)
            self.grasp_manager.mark_released(obj)

        # Set gripper to open position (no physics)
        self.set_kinematic_open()

    def _detect_grasp_geometric(self) -> str | None:
        """Detect grasp using geometric proximity (no physics contacts).

        Checks if any candidate object's center is close enough to the
        gripper position to be considered grasped.

        Returns:
            Name of closest object within threshold, or None
        """
        if not self._candidate_objects:
            return None

        # Get gripper position
        gripper_pos = self._get_gripper_position()
        if gripper_pos is None:
            return None

        closest_obj = None
        closest_dist = float("inf")

        for obj_name in self._candidate_objects:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
            )
            if body_id == -1:
                continue

            obj_pos = self.data.xpos[body_id]
            dist = np.linalg.norm(obj_pos - gripper_pos)

            if dist < self.kinematic_grasp_threshold and dist < closest_dist:
                closest_obj = obj_name
                closest_dist = dist

        return closest_obj

    def _get_gripper_position(self) -> np.ndarray | None:
        """Get the gripper's center position.

        Uses gripper site if available, otherwise averages gripper body positions.

        Returns:
            3D position array, or None if not available
        """
        # Try site first
        if self.gripper_site_name:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.gripper_site_name
            )
            if site_id != -1:
                return self.data.site_xpos[site_id].copy()

        # Fall back to average of gripper body positions
        if not self.gripper_body_names:
            return None

        positions = []
        for body_name in self.gripper_body_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id != -1:
                positions.append(self.data.xpos[body_id])

        if not positions:
            return None

        return np.mean(positions, axis=0)
