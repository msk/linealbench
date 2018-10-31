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
    let ddot_rblas = Fun::new("RBLAS", move |b, &i| {
        b.iter(|| rblas::Dot::dot(&xtmp[..i], &ytmp[..i]))
    });

    let xtmp = x.clone();
    let ytmp = y.clone();
    let ddot_nalgebra = Fun::new("nalgebra", move |b, &i| {
        let x = nalgebra::DVector::from_iterator(i, xtmp[..i].iter().cloned());
        let y = nalgebra::DVector::from_iterator(i, ytmp[..i].iter().cloned());
        b.iter(|| x.dot(&y))
    });

    let xtmp = x.clone();
    let ytmp = y.clone();
    let ddot_ndarray = Fun::new("ndarray", move |b, &i| {
        let x = ndarray::arr1(&xtmp[..i]);
        let y = ndarray::arr1(&ytmp[..i]);
        b.iter(|| x.dot(&y))
    });

    let ddot_funs = vec![ddot_rblas, ddot_nalgebra, ddot_ndarray];
    c.bench_functions("ddot", ddot_funs, 100);
}

criterion_group!(vector, ddot);
criterion_main!(vector);
