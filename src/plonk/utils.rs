use crate::pairing::ff::PrimeField;
use crate::worker::Worker;

pub(crate) fn convert_to_field_elements<F: PrimeField>(
    indexes: &[usize],
    worker: &Worker,
) -> Vec<F> {
    let mut result = vec![F::zero(); indexes.len()];

    worker.scope(indexes.len(), |scope, chunk| {
        for (idx, fe) in indexes.chunks(chunk).zip(result.chunks_mut(chunk)) {
            scope.spawn(move |_| {
                let mut repr = F::zero().into_repr();
                for (idx, fe) in idx.iter().zip(fe.iter_mut()) {
                    repr.as_mut()[0] = *idx as u64;
                    *fe = F::from_repr(repr).expect("is a valid representation");
                }
            });
        }
    });

    result
}
// b copy from a
// b[..(a.len())] = a[..]
pub(crate) fn fast_clone<F: PrimeField>(a: &[F], b: &mut [F], worker: &Worker) {
    let size = a.len();
    assert!(b.len() >= size, "The size of b should be greater than a.");

    let r = &mut b[..] as *mut [F];
    worker.in_place_scope(size, |scope, chunk| {
        for (i, v) in a.chunks(chunk).enumerate() {
            let r = unsafe { &mut *r };
            scope.spawn(move |_| {
                let start = i * chunk;
                let end = if start + chunk <= size {
                    start + chunk
                } else {
                    size
                };
                let copy_start_pointer: *mut F = r[start..end].as_mut_ptr();

                unsafe {
                    std::ptr::copy_nonoverlapping(v.as_ptr(), copy_start_pointer, end - start)
                };
            });
        }
    });
}

pub(crate) fn fast_initialize_to_element<F: PrimeField>(
    size: usize,
    initial_value: F,
    worker: &Worker,
) -> Vec<F> {
    if size == 0 {
        return vec![];
    }
    let mut a: Vec<F> = Vec::with_capacity(size);
    unsafe {
        a.set_len(size);
    }

    worker.in_place_scope(size, |scope, chunk| {
        for els in a.chunks_mut(chunk) {
            scope.spawn(move |_| {
                for mut el in els.iter_mut() {
                    *el = initial_value;
                }
            });
        }
    });

    a
}
