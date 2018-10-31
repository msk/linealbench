#[macro_use]
extern crate criterion;
extern crate nalgebra;
extern crate rand;

use criterion::Criterion;
use rand::distributions::Standard;
use rand::{IsaacRng, Rng, SeedableRng};

fn ddot(c: &mut Criterion) {
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();

    let xtmp = x.clone();
    let ytmp = y.clone();
    c.bench_function("ddot_rblas", move |b| {
        b.iter(|| rblas::Dot::dot(&xtmp, &ytmp))
    });

    let xtmp = nalgebra::DVector::from_iterator(x.len(), x.iter().cloned());
    let ytmp = nalgebra::DVector::from_iterator(x.len(), x.iter().cloned());
    c.bench_function("ddot_nalgebra", move |b| b.iter(|| xtmp.dot(&ytmp)));
}

criterion_group!(vector, ddot);
criterion_main!(vector);
