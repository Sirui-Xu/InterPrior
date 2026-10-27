"""Isaac Gym buffers and stepping for inference."""

from isaacgym import gymapi
import torch


class BaseTask:
    def __init__(self, cfg, enable_camera_sensors=False):
        self.gym = gymapi.acquire_gym()
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = int(cfg.get("device_id", 0))
        self.device = (
            "cuda:" + str(self.device_id)
            if self.device_type in ("cuda", "GPU")
            else "cpu"
        )
        self.headless = cfg["headless"]
        self.graphics_device_id = (
            self.device_id
            if cfg["env"].get("render_cameras", False) or not self.headless
            else -1
        )
        self.num_envs = int(cfg["env"]["numEnvs"])
        self.num_obs_teacher = int(cfg["env"]["numObsTeacher"])
        self.num_obs = self.num_obs_student = int(cfg["env"]["numObsStudent"])
        self.num_states = int(cfg["env"].get("numStates", 0))
        self.num_actions = int(cfg["env"]["numActions"])
        self.control_freq_inv = int(cfg["env"].get("controlFrequencyInv", 1))
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.obs_buf_raw = torch.zeros(
            (self.num_envs, self.num_obs_teacher), device=self.device
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf_count = torch.ones_like(self.reset_buf)
        self.progress_buf = torch.zeros_like(self.reset_buf)
        self.start_times = torch.zeros_like(self.reset_buf)
        self.randomize_buf = torch.zeros_like(self.reset_buf)
        self.contact_reset = torch.zeros(self.num_envs, device=self.device)
        self.data_id = torch.zeros_like(self.reset_buf)
        self.dataset_id = torch.zeros_like(self.reset_buf)
        self.extras = {}
        self.dr_randomizations = {}
        self.viewer = None
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError(
                    "Viewer creation failed; use --headless on a displayless machine"
                )

    def set_sim_params_up_axis(self, params, axis):
        if axis != "z":
            raise ValueError("This character uses Z-up coordinates")
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        return 2

    def create_sim(self, compute_device, graphics_device, physics_engine, params):
        sim = self.gym.create_sim(
            compute_device, graphics_device, physics_engine, params
        )
        if sim is None:
            raise RuntimeError("Isaac Gym simulator creation failed")
        return sim

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

    def render(self, sync_frame_time=False):
        if self.viewer is None:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            raise KeyboardInterrupt
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

    def get_states(self):
        return self.states_buf

    def apply_randomizations(self, *args, **kwargs):
        raise RuntimeError("This inference driver requires task.randomize=false")
