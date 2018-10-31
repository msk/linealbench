#[macro_use]
extern crate criterion;
extern crate nalgebra;
extern crate ndarray;
extern crate rand;

use criterion::{Criterion, Fun};
use rand::distributions::Standard;
use rand::{IsaacRng, Rng, SeedableRng};

fn ddot(c: &mut Criterion) {
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();

    let xtmp = x.clone();
    let ytmp = y.clone();
    let ddot_rblas = Fun::new("RBLAS", move |b, _| {
        b.iter(|| rblas::Dot::dot(&xtmp, &ytmp))
    });

    let xtmp = nalgebra::DVector::from_iterator(x.len(), x.iter().cloned());
    let ytmp = nalgebra::DVector::from_iterator(x.len(), x.iter().cloned());
    let ddot_nalgebra = Fun::new("nalgebra", move |b, _| b.iter(|| xtmp.dot(&ytmp)));

    let xtmp = ndarray::arr1(&x);
    let ytmp = ndarray::arr1(&y);
    let ddot_ndarray = Fun::new("ndarray", move |b, _| b.iter(|| xtmp.dot(&ytmp)));

    let ddot_funs = vec![ddot_rblas, ddot_nalgebra, ddot_ndarray];
    c.bench_functions("ddot", ddot_funs, 0);
}

criterion_group!(vector, ddot);
criterion_main!(vector);
