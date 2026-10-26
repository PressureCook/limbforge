"""Run the package's built-in end-to-end self-tests under pytest."""

from prosthetic_pipeline import selftest, selftest_pose


def test_geometry():
    selftest.test_geometry()


def test_limb_demo_data():
    selftest.test_limb_demo()


def test_full_chain_evolution_actuator_stl_usage():
    m, final, ctx = selftest.test_evolution_and_sim()
    selftest.test_actuator_design(m, final, ctx)
    selftest.test_printable(m, final, ctx)
    selftest.test_usage_tasks(m, ctx)


def test_pose_extraction_with_stubbed_detector():
    selftest_pose.main()
