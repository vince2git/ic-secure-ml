<script>
    import { onMount } from 'svelte';

    let showBanner = true;


    onMount(() => {
        // Check if consent has been previously given
        const storedConsent = localStorage.getItem('matomoConsent');
        if (storedConsent === 'true') {
            showBanner = false;
        }
    });

    function giveConsent() {
        showBanner = false;
        localStorage.setItem('matomoConsent', 'true');
    }




</script>
{#if showBanner}
<div class="consent-banner">
    <p>
        We use Matomo analytics to improve and benchmark this project. This does not involve any personal data collection.
    </p>
    <div class="button-group">
        <button on:click={giveConsent} class="accept">OK</button>

    </div>
</div>
{/if}

<style>
    .consent-banner {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        background-color: #f3f4f6;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        font-family: Arial, sans-serif;
        width: 35%;
    }

    p {
        margin: 0 0 1rem 0;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .button-group {
        display: flex;
        gap: 0.5rem;
        float: right;
    }

    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background-color 0.3s ease;
        width: 100px;
        background-color: #a5d4d4;
    }




</style>