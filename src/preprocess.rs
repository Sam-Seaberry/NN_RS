use nalgebra::DMatrix;
use ndarray::Array2;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::s;


#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}
#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub classification: Vec<f32>,
}

impl MnistData {
    fn new(mut f: &File) -> Result<MnistData, std::io::Error> {
        //let mut gz = GzDecoder::new(f);
        //let mut contents: Vec<u8> = Vec::new();
        //gz.read_to_end(&mut contents)?;
        let mut contents: Vec<u8> = Vec::new();
        f.read_to_end(&mut contents)?;
        let mut r = Cursor::new(contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}-labels.idx1-ubyte", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}-images.idx3-ubyte", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Array2<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(Array2::from_shape_vec((image_shape, 1), image_data).unwrap());
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        //simplifing training on only two digits 1 and 0
        if classification == 1 | 0 {
            ret.push(MnistImage {
                image,
                classification: value_to_vec(classification as f32),
            });
        }
    }

    Ok(ret)
}

pub fn create_ndmatrix_from_mnist_image(image: &MnistImage, shape: Vec<usize>) -> DMatrix<f32> {
    let num_rows = shape[0];
    let num_cols = shape[1];

    let image_vec = image.image.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let mut rows:Vec<Vec<f32>> = Vec::new();

    /*for i in 0..num_rows{
        let row = (0..num_cols).map(|x| image.image[x] as f32).collect();
        rows.push(row);
    }*/

    //slet flat_data:Vec<f32> = rows.iter().flat_map(|r| r.iter()).copied().collect();

    let matrix = DMatrix::from_row_slice(num_rows as usize, num_cols as usize, &image_vec);

    matrix

}

fn value_to_vec(value: f32)->Vec<f32>{
    let mut ret:Vec<f32> = Vec::new();
    for i in 0..(value as i32){
        ret.push(0.0);
    }
    ret.push(1.0);
    if value != 9.0{
        for i in (value as i32)..9{
            ret.push(0.0);
        }
    }
    ret
}