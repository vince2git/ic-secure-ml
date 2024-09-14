/* tslint:disable */
/* eslint-disable */
/**
*/
export class NeuralNetClient {
  free(): void;
/**
* @returns {NeuralNetClient}
*/
  static new(): NeuralNetClient;
/**
* @param {any} input
* @returns {any}
*/
  forward(input: any): any;
/**
* @param {any} gradients
*/
  backward(gradients: any): void;
}
