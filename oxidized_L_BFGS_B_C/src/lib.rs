mod lbfgsb_bind;
use libc::{c_char, c_double, c_long};

fn step(n:c_long,m:c_long,x:&mut [c_double],l:&[c_double],u:&[c_double],nbd:&[c_long],f:c_double,g:&[c_double],factr:c_double,pgtol:c_double,wa:&mut [c_double],iwa:&mut [c_long],task:&mut [c_long],iprint:c_long,csave:&mut [i8],lsave:&mut [c_long],isave:&mut [c_long],dsave:&mut [c_double]){
    unsafe{
           lbfgsb_bind::setulb(&n,&m,x.as_mut_ptr(),l.as_ptr(),u.as_ptr(),nbd.as_ptr(),&f,g.as_ptr(),&factr,&pgtol,wa.as_mut_ptr(),
           iwa.as_mut_ptr(),task.as_mut_ptr(),&iprint,csave.as_mut_ptr(),lsave.as_mut_ptr(),isave.as_mut_ptr(),dsave.as_mut_ptr())
    }
}
#[derive(Debug)]
pub enum ConvergenceTypes {
    PgtolReached,
    TermCondReached,
    LinesearchExhausted,
    MaxItersReached
}

pub struct Lbfgsb<'a>{
    n:c_long,
    m:c_long,
    x:&'a mut Vec<c_double>,
    l:Vec<c_double>,
    u:Vec<c_double>,
    nbd:Vec<c_long>,
    f:&'a dyn Fn(&Vec<c_double>)->c_double,
    g:&'a dyn Fn(&Vec<c_double>)->Vec<c_double>,
    factr:c_double,
    pgtol:c_double,
    wa:Vec<c_double>,
    iwa:Vec<c_long>,
    task:Vec<c_long>,
    iprint: c_long,
    csave: Vec<c_char>,
    lsave: Vec<c_long>,
    isave:Vec<c_long>,
    dsave:Vec<c_double>,
    max_iter: u32
}
#[allow(dead_code, unused_assignments)]
impl<'a> Lbfgsb<'a>{
    //constructor requres three mendatory parameter which is the initial solution, function and the gradient function
    pub fn new(xvec:&'a mut Vec<c_double>,func:&'a dyn Fn(&Vec<c_double>)->c_double,gd:&'a dyn Fn(&Vec<c_double>)->Vec<c_double>)->Self{
        let len = xvec.len() as c_long;
        //creating lbfgs struct
        Lbfgsb{n:len,m:5,x:xvec,l:vec![0.0f64;len as usize],u:vec![0.0f64;len as usize],nbd:vec![0;len as usize],f:func,g:gd,
             factr:0.0e0,pgtol:0.0e0,wa:vec![0.0f64;(2*5*len+11*5*5+5*len+8*5) as usize],iwa:vec![0;3*len as usize],task:vec![1],iprint:-1,csave:vec![0;60],
             lsave:vec![0,0,0,0],isave:vec![0;44],dsave:vec![0.0f64;29],max_iter:0
        }
    }
    //this function will start the optimization algorithm
    pub fn minimize(&mut self) -> Option<ConvergenceTypes>{
        let mut fval = 0.0f64;
        let mut gval = vec![0.0f64;self.x.len()];
        let func = self.f;
        let grad = self.g;
       //start of the loop
       loop{
           //callign the fortran routine
           step(self.n,self.m,&mut self.x,&self.l,&self.u,&self.nbd,fval,&gval,self.factr,self.pgtol,&mut self.wa,
                 &mut self.iwa,&mut self.task,self.iprint,&mut self.csave,&mut self.lsave,&mut self.isave,&mut self.dsave);

            // Task numbers are defined as constants in lbfgsb.h, task variable is updated after every step
            
            if self.task[0] >= 10 && self.task[0] <= 15 { //New point requested in linesearch
                fval = func(self.x);
                gval = grad(self.x);
            }

            if self.task[0]==2 { //iteration complete
                continue;
            }
            
            if self.max_iter>0 && self.isave[29]>= self.max_iter as c_long {
                return Some(ConvergenceTypes::MaxItersReached)
            }
            
            if self.task[0] == 21 {
                return Some(ConvergenceTypes::PgtolReached);
            }
            
            if self.task[0] == 22 {
                return Some(ConvergenceTypes::TermCondReached);
            }

            if self.task[0] == 3{
                return Some(ConvergenceTypes::LinesearchExhausted);
            }

            if self.task[0] >= 200 && self.task[0] <= 240{
                return None
            }
        }
    }
    //this function returns an owned solution after minimization
    pub fn get_x(&self)->Vec<c_double>{
        self.x.clone()
    }
    //this function is used to set lower bounds to a variable
    pub fn set_lower_bound(&mut self,index:usize,value:f64){
        if self.nbd[index]==1 || self.nbd[index]==2{
            println!("variable already has Lower Bound");
        }
        else{
            let temp = self.nbd[index]-1;
            self.nbd[index] = if temp<0{-temp}else{temp};
            self.l[index] = value;
        }
    }
    //this function is used to set upper bounds to a variable
    pub fn set_upper_bound(&mut self,index:usize,value:f64){
        if self.nbd[index]==3 || self.nbd[index]==2{
            println!("variable already has Lower Bound");
        }
        else{
            self.nbd[index] = 3-self.nbd[index];
            self.u[index] = value;
        }
    }
    //set the verbosity level
    pub fn set_verbosity(&mut self,l:i32){
        self.iprint = l.into();
    }
    //set termination tolerance
        //1.0e12 for low accuracy
        //1.0e7  for moderate accuracy
        //1.0e1  for extremely high accuracy
    pub fn set_termination_tolerance(&mut self,t:f64){
        self.factr = t;
    }
    //set tolerance of projection gradient
    pub fn set_pgtolerance(&mut self,t:f64){
        self.pgtol = t;
    }
    //set max iteration
    pub fn max_iteration(&mut self,i:u32){
        self.max_iter = i;
    }
    //set maximum number of variable metric corrections
    //The range  3 <= m <= 20 is recommended
    pub fn set_matric_correction(&mut self,m:i32){
        self.m = m.into();
    }
}

