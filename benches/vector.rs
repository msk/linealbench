#[macro_use]
extern crate criterion;
extern crate nalgebra;
extern crate ndarray;
extern crate rand;

use criterion::{Criterion, ParameterizedBenchmark};
use rand::distributions::Standard;
use rand::{IsaacRng, Rng, SeedableRng};

fn ddot(c: &mut Criterion) {
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();

    let params = vec![10, 500, 1000];
    let (x_rblas, y_rblas) = (x.clone(), y.clone());
    let (x_nalgebra, y_nalgebra) = (x.clone(), y.clone());
    let (x_ndarray, y_ndarray) = (x.clone(), y.clone());

    let benchmark = ParameterizedBenchmark::new(
        "RBLAS",
        move |b, &i| b.iter(|| rblas::Dot::dot(&x_rblas[..i], &y_rblas[..i])),
        params,
    ).with_function("nalgebra", move |b, &i| {
        let x = nalgebra::DVector::from_iterator(i, x_nalgebra[..i].iter().cloned());
        let y = nalgebra::DVector::from_iterator(i, y_nalgebra[..i].iter().cloned());
        b.iter(|| x.dot(&y))
    }).with_function("ndarray", move |b, &i| {
        let x = ndarray::arr1(&x_ndarray[..i]);
        let y = ndarray::arr1(&y_ndarray[..i]);
        b.iter(|| x.dot(&y))
    });
    c.bench("ddot", benchmark);
}

criterion_group!(vector, ddot);
criterion_main!(vector);
