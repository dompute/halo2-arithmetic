use std::os::raw::c_void;

#[repr(C)]
#[derive(Debug)]
pub struct U256([u64; 4]);

#[repr(C)]
pub enum CalculationTag {
    Add,
    Sub,
    Mul,
    Square,
    Double,
    Negate,
    Horner,
    Store,
}

extern "C" {
    pub fn create_graph() -> *const c_void;
    pub fn delete_graph(graph: *const c_void);
    pub fn reset_graph(graph: *const c_void, num: usize);

    pub fn push_node(
        graph: *const c_void,
        target: usize,
        tag: CalculationTag,
        vs: *const c_void,
        vs_len: usize,
    );
    pub fn evaluate_batch(
        values: *mut U256,
        values_len: usize,
        graph: *const c_void,
        rotations: *const i32,
        rotations_len: usize,
        constants: *const U256,
        constants_len: usize,
        fixed: *const *const U256,
        fixed_col: usize,
        fixed_row: usize,
        advice: *const *const U256,
        advice_col: usize,
        advice_row: usize,
        instance: *const *const U256,
        instance_col: usize,
        instance_row: usize,
        challenges: *const U256,
        challenges_len: usize,
        beta: *const U256,
        gamma: *const U256,
        theta: *const U256,
        y: *const U256,
        rot_scale: i32,
        isize: i32,
    );

    pub fn msm_fr_g1(bases: *const c_void, scalars: *const c_void, buf_len: u32, out: *mut c_void);

    pub fn fft_fr(values: *mut U256, twiddles: *const U256, log_n: u32);
}
