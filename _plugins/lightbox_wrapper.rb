# _plugins/lightbox_wrapper.rb
# 自动给文章内容中的 <img> 包裹 <a> 链接指向大图

require 'nokogiri'

module Jekyll
  module LightboxWrapper
    Jekyll::Hooks.register :posts, :post_render do |post|
      next unless post.output_ext == ".html"

      doc = Nokogiri::HTML::DocumentFragment.parse(post.output)
      doc.css('img').each do |img|
        # 如果 img 已经被 a 标签包裹则跳过
        next if img.parent.name == 'a'

        # 创建 a 标签包裹 img
        a = Nokogiri::XML::Node.new "a", doc
        a['href'] = img['src']
        a['class'] = "lightbox"
        img.replace(a)
        a.add_child(img)
      end

      post.output = doc.to_html
    end
  end
end
