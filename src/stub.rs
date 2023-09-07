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

    pub fn push_node(
        graph: *const c_void,
        target: usize,
        tag: CalculationTag,
        vs: *const c_void,
        vs_len: usize,
    );
    pub fn evaluate_batch(
        values: *mut c_void,
        values_len: usize,
        graph: *const c_void,
        rotations: *const i32,
        rotations_len: usize,
        constants: *const c_void,
        constants_len: usize,
        fixed: *const *const c_void,
        fixed_col: usize,
        fixed_row: usize,
        advice: *const *const c_void,
        advice_col: usize,
        advice_row: usize,
        instance: *const *const c_void,
        instance_col: usize,
        instance_row: usize,
        challenges: *const c_void,
        challenges_len: usize,
        beta: *const c_void,
        gamma: *const c_void,
        theta: *const c_void,
        y: *const c_void,
        rot_scale: i32,
        isize: i32,
        round: usize,
    );

    pub fn msm_fr_g1(scalars: *const c_void, bases: *const c_void, buf_len: u32, out: *mut c_void);

    pub fn fft_fr(values: *mut c_void, twiddles: *const c_void, log_n: u32);
}
