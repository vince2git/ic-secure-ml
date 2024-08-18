<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  const presets = [
    { name: "Fast (near query limit)", pca: 10, perp: 10, thres: 0.9, iter: 100 },
    { name: "Balanced", pca: 50, perp: 50, thres: 0.5, iter: 100 },
    { name: "Without Barnes and Hut optimisation (near update limit)", pca: 50, perp: 50, thres: 0, iter: 200 }
  ];

  let sliderValue = 0;

  $: currentPreset = presets[sliderValue];

  $: {
    dispatch('configChange', currentPreset);
  }
</script>

<div class="space-y-6 quick">
  <div class="flex justify-between mb-2">
      Quick parameters setting
  </div>
  <input
    type="range"
    min="0"
    max="2"
    step="1"
    bind:value={sliderValue}
    class="w-full"
  />
  <div class="text-center mt-4">
    <span class="font-medium">Current : </span>
    {currentPreset.name}
  </div>

</div>

<style>
      .quick{
  background-color: #e8f4ff;
}
  input[type="range"] {
    -webkit-appearance: none;
    width: 80%;
    height: 15px;
    border-radius: 5px;
    background: #d3d3d3;
    outline: none;
    opacity: 0.7;
    transition: opacity .2s;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 25px;
    height: 25px;
    border-radius: 50%;
    background: #4CAF50;
    cursor: pointer;
  }

  input[type="range"]::-moz-range-thumb {
    width: 25px;
    height: 25px;
    border-radius: 50%;
    background: #4CAF50;
    cursor: pointer;
  }
</style>
