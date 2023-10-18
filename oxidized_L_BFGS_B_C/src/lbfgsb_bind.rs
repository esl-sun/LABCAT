use libc::{c_double,c_long,c_char};

extern "C"{
    pub fn setulb(n:*const c_long,m:*const c_long,x:*mut c_double,l:*const c_double,
    u:*const c_double,nbd:*const c_long,f:*const c_double,g:*const c_double,factr:*const c_double,
    pgtol:*const c_double,wa:*mut c_double,iwa:*mut c_long,task:*mut c_long,iprint:*const c_long,csave:*mut c_char,
    lsave:*mut c_long,isave:*mut c_long,dsave:*mut c_double);
}