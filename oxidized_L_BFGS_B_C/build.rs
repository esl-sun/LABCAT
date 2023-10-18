use cc;

fn main() {
    cc::Build::new()
        .file("src/c_source/lbfgsb.c")
        .file("src/c_source/linesearch.c")
        .file("src/c_source/subalgorithms.c")
        .file("src/c_source/print.c")
        .file("src/c_source/linpack.c")
        .file("src/c_source/miniCBLAS.c")
        .file("src/c_source/timer.c")
        .warnings(false)
        .compile("lbfgsb_c");
}