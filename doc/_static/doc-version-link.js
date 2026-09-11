// Point the links in the announcement bar at the sibling documentation tree.
//
// The bar is one HTML string shared by every page, so a relative path would resolve differently
// depending on how deep the page is. Sphinx renders rel="index" relative to the tree root on every
// page, so resolving it gives the root, and the sibling tree is one level up from there. That keeps
// this independent of the domain and of the path prefix the site is served under.
window.addEventListener("DOMContentLoaded", function () {
    var index = document.querySelector('link[rel="index"]');
    if (!index) {
        return;  // no index to anchor on, leave the fallback href in place
    }
    var root = new URL("./", new URL(index.getAttribute("href"), window.location.href));
    document.querySelectorAll("a[data-doc-tree]").forEach(function (link) {
        link.href = new URL("../" + link.dataset.docTree + "/", root).href;
    });
});
