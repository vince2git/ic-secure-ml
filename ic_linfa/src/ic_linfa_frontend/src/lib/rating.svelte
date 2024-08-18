<script>
  let rating = 0;
  let hover = 0;
  let showThankYou = false;

  function handleRating(selectedRating) {
    rating = selectedRating;
    fetch('https://example?idsite=1&rec=1&_cvar={"1":["rating","'+selectedRating+'"]}',{
      method: 'GET',
      mode: 'no-cors'});
    showThankYou = true;
  }
  function handleKey(event,val) {
    if(event.enter)
      handleRating(val);
  }

  function handleMouseEnter(value) {
    hover = value;
  }

  function handleMouseLeave() {
    hover = 0;
  }
</script>

<div class="star-rating-popup">
  <h2>Rate this project 📝</h2>
  <div class="stars">
    {#each Array(5) as _, i}
      {@const starValue = i + 1}
      <svg
        class="star {starValue <= (hover || rating) ? 'active' : ''}"
        width="32"
        height="32"
        viewBox="0 0 24 24"
        role="link"
        tabindex="{starValue}"
        aria-roledescription="rate the app"
        on:keyup={() => handleKey(starValue)}
        on:click={() => handleRating(starValue)}
        on:mouseenter={() => handleMouseEnter(starValue)}
        on:mouseleave={handleMouseLeave}
      >
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
      </svg>
    {/each}
  </div>
  {#if showThankYou}
    <div class="thank-you-message">
      <p><strong>Thank you for your feedback!</strong></p>
      <p>You can find the <a href="https://github.com/vince2git/ic-secure-ml">source code here</a></p>
    </div>
  {/if}
</div>

<style>
  .star-rating-popup {
    padding: 1rem;
    background-color: white;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    font-family: Arial, sans-serif;
  }

  h2 {
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 1rem;
  }

  .stars {
    display: flex;
    margin-bottom: 1rem;
  }

  .star {
    cursor: pointer;
    fill: none;
    stroke: #d1d5db;
    stroke-width: 2;
    transition: all 0.2s ease;
  }

  .star.active {
    fill: #fbbf24;
    stroke: #fbbf24;
  }

  .thank-you-message {
    margin-top: 1rem;
    padding: 1rem;
    background-color: #e0f2fe;
    border-left: 4px solid #3b82f6;
    color: #1e40af;
  }

  .thank-you-message p {
    margin: 0;
  }

  .thank-you-message p + p {
    margin-top: 0.5rem;
  }
</style>