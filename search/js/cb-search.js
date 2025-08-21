$(function () {
    var names = [];
    var urls = [];

    // 定义匹配函数
    function substringMatcher(strs) {
        return function findMatches(q, cb) {
            var matches = [];
            var substrRegex = new RegExp(q, "i");
            $.each(strs, function (i, str) {
                if (substrRegex.test(str)) {
                    matches.push(str);
                }
            });
            cb(matches);
        };
    }

    // 自动拼接 baseurl
    var baseurl = "{{ site.baseurl }}";
    if (!baseurl.endsWith("/")) {
        baseurl += "/";
    }
    var searchPath = baseurl + "search/cb-search.json";

    $.getJSON(searchPath)
        .done(function (data) {
            if (data.code === 0 && data.data && data.data.length > 0) {
                data.data.forEach(function (item) {
                    names.push(item.title);
                    urls.push(item.url);
                });

                // 初始化 typeahead
                $("#cb-search-content").typeahead(
                    {
                        hint: true,
                        highlight: true,
                        minLength: 1
                    },
                    {
                        name: "posts",
                        source: substringMatcher(names)
                    }
                );

                // 监听选择
                $("#cb-search-content").bind("typeahead:select", function (ev, suggestion) {
                    var idx = names.indexOf(suggestion);
                    if (idx !== -1) {
                        window.location.href = urls[idx].trim();
                    }
                });
            }
        })
        .fail(function (jqxhr, textStatus, error) {
            console.error("加载 cb-search.json 出错:", textStatus, error);
        });
});
